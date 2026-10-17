# Phase A — Research report: reference projects and the decisions they force

Inspection date: 2026-10-12. Reference archives inspected read-only in place; working copies
extracted to a local research directory outside this repository (`openwhispr-main/`,
`pocket-tts-main/`). The original ZIPs were
not modified.

Evidence classes used throughout, kept deliberately separate:

- **[D]** documented claim by the project itself (README, docs, model card).
- **[C]** verified by reading the code in the archive.
- **[M]** measured data present in the archive (a benchmark artefact with per-run samples).
- **[N]** not found — absence verified by inspection, not inferred from silence.

No capability below is asserted on the basis of a project name or marketing line.

---

## 1. OpenWhispr — what it actually is

Version 1.10.3, changelog dated 2026-10-09, licence MIT (`LICENSE:1`). It is not a dictation demo:
it is an Electron desktop product with hotkey dictation at the cursor, meeting transcription with
live diarization, notes plus local semantic search, an agent/voice assistant, audio and YouTube
import, a public API and MCP server, and an Expo mobile app **[D]** `README.md:16-30`,
`package.json:3`. Platform boundary is desktop only (macOS arm64+x86_64, Windows, Linux) **[C]**.

Publisher is "Gizmo Labs Inc." **[C]** `electron-builder.json:188`, and "OpenWhispr Cloud" is a
first-party hosted service, so cloud dependency is structural rather than incidental. This matters
for this project: the reference is a hybrid product, not a local-first reference, and its defaults reflect
that (`useLocalWhisper` default **false** **[C]** `src/stores/settingsStore.ts:1503-1506`).

### 1.1 Recognition engines actually present

Six families **[C]**, all resolvable to code:

| Family | Models | Source | Cache location |
|---|---|---|---|
| whisper.cpp (GGML) | tiny → large-v3, default `base` | own fork `OpenWhispr/whisper.cpp` pinned tag `0.0.10` | `~/.cache/openwhispr/whisper-models` |
| sherpa-onnx INT8 | Parakeet TDT 0.6B v3, Parakeet Unified EN, Nemotron Streaming EN, Nemotron 3.5 ASR, Cohere Transcribe 2B, Orukeet | `k2-fsa` releases + HF | `~/.cache/openwhispr/parakeet-models` |
| cloud batch | OpenAI, Groq, xAI, Mistral Voxtral, Gemini, Corti, Tinfoil, Deepgram, AssemblyAI (9) | APIs | n/a |
| cloud streaming | Deepgram Nova-3, OpenAI Realtime, AssemblyAI, Gemini Live, managed Orukeet | APIs | n/a |
| self-hosted | OpenAI-compatible `/transcriptions` + a documented custom ASR shim | user | n/a |

`scripts/download-whisper-cpp.js:19-23`, `src/models/modelRegistryData.json`,
`src/helpers/parakeetServer.js`, `CLAUDE.md:249-275`.

Model sizes **[C]**: Parakeet ~680 MB, Cohere 1.7 GB download / 2.7 GB on disk, Orukeet 672 MB.
Whisper documented RAM **[D]** `LOCAL_WHISPER_SETUP.md:17-24`: tiny ~1 GB → large ~10 GB.
Cohere claims 5.42 % average WER across 14 languages, Parakeet Unified EN 5.91 % **[D]**, quoted into
the registry at `CLAUDE.md:263,266` — third-party claims, not their measurements.

### 1.2 Capture and preprocessing — the load-bearing decisions

- Microphone is opened **stereo with OS/WebRTC processing disabled**: `channelCount: 2`,
  `echoCancellation/noiseSuppression/autoGainControl` all false **[C]** `src/helpers/audioManager.js:1104-1109`.
  Reasons are recorded in comments: mono WebM breaks silence detection on PipeWire (#472), and
  Chromium AGC mutates the Windows system mic volume (#476). *Ship raw, gate in your own code.*
- A 16 kHz mono PCM16 AudioWorklet tap runs beside the MediaRecorder Opus stream, so offline engines
  need no post-stop FFmpeg decode **[C]** `src/helpers/pcmTap.js:4,55-58`. The tap self-vetoes: 240 s cap,
  1 s flush watchdog, and rejection if the copy is >100 ms shorter than the recording.
- No DC removal, high-pass or gain normalisation anywhere in the pipeline **[N]** (searched).
- System-audio meeting capture is 24 kHz mono s16le in 100 ms chunks **[C]** `audioTapManager.js:11-13`.
  AEC is a separate native helper using WebRTC AEC3 at 48 kHz internally, **opt-in and off by default**
  because AEC3 also removed the near-end speaker's own speech, worst in the first seconds (#1025) **[D]** `CLAUDE.md:768`.

### 1.3 Voice activity detection — three layers, and a cautionary tale

1. Silero v5.1.2 through whisper.cpp, defaults threshold 0.5, minSpeech 250 ms, minSilence 200 ms,
   maxSpeech 30 s, speechPad 100 ms **[C]** `src/constants/whisperVad.json`, `scripts/download-whisper-vad-model.js:6-8`.
2. An energy gate in JS: silence RMS 0.002, speech-window RMS 0.003 + peak 0.02, strong RMS 0.006 **[C]** `localSpeechGate.js:1-4`.
3. Silero ONNX + pyannote segmentation for meetings/diarization, min-duration-on 0.2 s / off 0.5 s **[C]** `diarization.js:305-315`.

The decision recorded at `whisperVadConfig.js:24-31` is the important one: **VAD is opt-in for
dictation but on by default for notes/meetings**, because applying an aggressive gate to pause-heavy
dictation deleted real speech, and Whisper then decoded the near-silence and emitted dictionary words
from its own prompt (#1454). Endpointing for dictation is hotkey-driven, not VAD-driven.

### 1.4 Streaming

Both, split by a `runtime: offline | online` field in the model registry **[C]**. Offline models decode
after stop (or in 1.5 s buffered preview chunks). Online models stream worklet PCM over a persistent
websocket, commit flushed text at stop, no second decode; a truncation-aware flush that is not clean
falls back to record-then-transcribe **[D]** `CLAUDE.md:270`. Partial results **replace** preview text
online and **append** offline. Meeting streams emit `partial` / `final` / `retract` events consumed by a
pure reducer doing timestamp-sorted insert and exact-match retraction **[C]** `CLAUDE.md:769-770`.
Segment ceilings: 15 s offline, 30 s Cohere (model max ~35 s), 600 s online **[C]** `parakeetServer.js:26-32`.

### 1.5 Language handling — the gap that matters most here

`src/config/languageRegistry.json` holds **61 entries: 60 languages + `auto`** **[C]**, with
per-engine support flags: 59 whisper, 26 parakeet, 7 assemblyai.

- **Hindi (`hi`): present**, whisper-true, parakeet-false; its registry instruction explicitly tells the
  cleanup LLM to preserve Hinglish code-switching **[C]**.
- **Bengali (`bn`): absent from the registry entirely.** The only occurrence in the whole repository is
  inside Deepgram Nova-3's internal fallback set (`src/helpers/deepgramStreaming.js:19`) — a cloud
  capability never surfaced in the picker **[C]**. Searched; no registry entry, no UI path.
- Auto-detection has two implementations: whisper.cpp `--language auto`, and for the managed Orukeet
  stream a **separate acoustic LID classifier** emitting `language`/`language_confidence` at 3 s and 6 s,
  documented as "not a calibrated probability" and "not a voice-activity or code-switching detector" **[D]** `docs/orukeet-streaming-language.md:19-23`.
- Measured LID **[M]** `docs/orukeet-streaming-language.md:26-32`, 330 FLEURS clips / 33 languages:
  **52.1 % at 2 s, 77.6 % at 3 s, 93.0 % at 6 s**; at threshold 0.90 it caught 62.5 % / 92 % of
  unsupported clips with 1.2 % / 0.4 % false flags. Routing policy is confidence-thresholded and
  cost-asymmetric: an out-of-set estimate ≥0.90 discards the stream text and re-transcribes by batch.
- Per-language ASR accuracy caveats: **[N]** — none documented for local models.
- Translation is an LLM prompt slot with `{{targetLanguage}}`, plus OpenCC script conversion for Chinese **[C]**.

### 1.6 Personalisation and speaker identity

Dictionary words and snippet triggers become ASR hints: joined as the Whisper `prompt`, and as provider
`keyterms` capped at 100 terms × 50 chars **[C]** `audioManager.js:3687-3689`; a large dictionary broke
`gpt-transcribe` around 1,000 parts, so it is now capped at 900 with the remainder moved into the prompt **[D]**.
Local diarization is real: pyannote segmentation + 3D-Speaker CAM++ embeddings, cluster threshold 0.55,
live identification speech 0.15 / silence 0.08 / embedding match 0.65, persisted per note in SQLite **[C]**.
No per-application profiles **[N]**.

### 1.7 Performance and hardware

Anti-hallucination thresholds `entropy_thold 2.8`, `logprob_thold −1.25` cut hallucinated tails from
**2.25 % → 0.06 % over 4,814 real dictations**, and are disabled for meeting chunks because the raised
entropy triggers up to ~6× temperature-fallback re-decode **[M]/[D]** `whisperServer.js:36-47`.
GPU: Metal built in; CUDA and Vulkan one-click packs from their own fork, SHA-256 verified, with
automatic CPU fallback on crash/OOM and the failing error line surfaced to the user **[C]** `gpuBinaryManager.js:50,147-154`.
Threads: default 4, auto = 75 % of parallelism capped at 12 **[C]**. CoreML **[N]**.
**No local-model RTF or latency measurement exists anywhere in the archive [M absent]** — the JSON
benchmark artefacts (`docs/integrations/orukeet/benchmark-results.json`, `live-adapter-results.json`)
measure their **cloud** endpoint: stop→final median 92.3 ms (p95 144.8 ms) on a California Mac, median
inference ~15.3 ms, 86.8–91.2 utterances/s/GPU saturated, 24,361 matching finals. They carry explicit
self-limiting disclaimers ("not a worldwide SLA"; "colocated cloud clients are not representative").

### 1.8 Offline story

Fully offline-capable engines exist (whisper.cpp server, all sherpa INT8 models, local diarization,
llama.cpp GGUF, Qdrant + MiniLM search) **[C]**. But enforcement is an operator firewall guide
(`docs/network-allowlist.md`), not an app mode; the only guard found is an error string
"You're offline. Cloud transcription requires an internet connection." **[C]** `audioManager.js:3472`.
The offline sherpa websocket server binds 127.0.0.1 but has no loopback-only option, so Windows shows a
firewall prompt **[D]** `TROUBLESHOOTING.md:222-224`.

---

## 2. Pocket TTS — what it actually is

Version 3.3.0, PyPI `pocket-tts`, Kyutai Labs, MIT licence text present but **without a
copyright-holder line and with no `license` field in `pyproject.toml`** **[C]** — a real ambiguity.
Architecture **[C]** `CONTRIBUTING.md:65-73`, `models/flow_lm.py`, `models/mimi.py`:

1. **Mimi VAE encoder** → continuous latents (dim 32) from reference audio.
2. **Text conditioner** is only a tokenizer + embedding lookup table (`LUTConditioner`, `n_bins=4000`)
   — there is no deep text network.
3. **FlowLM**: a streaming transformer (`d_model=1024, heads=16, layers=6` default) generating one
   continuous latent per frame via flow matching / Lagrangian Self Distillation; three samplers
   (`lsd` 1-step, `flow_matching` ≥16 steps, `drifting` 1-step time-free). EOS is a separate linear head.
4. **Mimi VAE decoder** → 24 kHz mono. Frame rate **12.5 Hz** (80 ms per frame).

Headline claims **[D]** `README.md:22-33`: ~100M parameters, CPU-only, ~200 ms first chunk, ~6× real-time
on a MacBook Air M4 CPU using 2 cores, voice cloning, 7 languages, unbounded input length.
Measured numbers **[M]** present in `docs/quantization.md:43-57`: RTT-benchmarked RTF **3.17× → 4.04×**
(x86) and **6.33× → 7.76×** (ARM) with dynamic int8 on attention+FFN only; runtime memory
**450 MB → 234 MB**; WER delta **−0.022 ± 0.032** which crosses zero, i.e. no measurable quality cost.

### 2.1 Voice conditioning

Zero-shot from reference audio, not fine-tuning and not a fixed preset list **[C]** `tts_model.py:921-1047`:
audio → Mimi encoder latents → linear speaker projection → fed as an audio-conditioning prefix, and the
resulting **voice is a KV-cache snapshot plus offsets**, exportable as safetensors
(`models/model_state.py:15-20`, `docs/CLI Commands/export_voice.md`). Only the **first 30 s** of a
reference is used. A 26-voice catalogue ships as precomputed states; passing a preset to a custom
checkpoint raises **[C]** `tts_model.py:992-1000`. Sample quality is reproduced, so the prompt must be
clean first **[D]** `README.md:92`. Impersonation without lawful consent is explicitly prohibited **[D]**.

### 2.2 Text processing

Grapheme-only: **no phonemizer, no G2P** — pronunciation is learned implicitly **[C]**. Tokenizer is
SentencePiece/HF `tokenizers`, 4000 subwords per language. Normalisation is minimal: whitespace
collapse, optional first-letter capitalisation, guaranteed terminal punctuation, semicolon→comma, and
per-language `replace_characters` (French strips quotes/parens/brackets, maps `:`→`,`) **[C]**
`text_chunking.py:15-79`, `config/french.yaml:4-19`. **No number or abbreviation expansion** — digits
pass through and mis-reading digits is a *tested* failure mode (`training/eval/release_checks.py:55-63`).
Chunking splits on sentence boundaries decimal-aware, sub-splits oversized sentences on `,;:`, regroups
to ≤50 tokens, and generates each chunk **independently** — the cross-chunk conditioning is an open
TODO **[C]** `tts_model.py:710-713`.

### 2.3 Languages

Official, shipped, documented: **7** — English, French, German, Portuguese, Italian, Spanish, Dutch
(+ dated English variants and `_24l` higher-quality previews explicitly labelled "not the final models").
**Bengali: not found** — no config, no docs, zero hits repo-wide **[N]**.
**Hindi: not officially supported** — exists only as a community checkpoint `hf://saryps-labs/pocket-tts-hindi`
loaded via `--config` **[D]** `README.md:306-314`; community models cannot use the preset voices.
So a ~60-language requirement means authoring ~53 new models, each needing 100–2000 h of aligned
speech+transcript data **[D]** `training/README.md:104-135`.

### 2.4 Training and evaluation exist

Contrary to the "inference-only demo" reading, the archive contains `training/train.py`, a dataloader,
Muon optimizer, prepare/align/precompute/tokenizer scripts, and configs for `scratch`, `depth_distill`,
`finetune`, `finetune_language`, `drifting` **[C]**. Two-stage recipe: 24-layer teacher → 6-layer
depth-distilled student with classifier-free guidance baked in. Evaluation is LibriSpeech test-clean
cross-sentence (F5-TTS protocol, 1127 items) reporting **WER** (ASR = `ibm-granite/granite-speech-4.1-2b`),
**speaker similarity** (WavLM x-vector cosine) and **UTMOS**; documented released-English numbers
**WER 0.90 %, sim 0.922, UTMOS 4.36** **[M]** `training/README.md:243-260`. Training requires Linux +
NVIDIA GPU (~56 GB batch default, ~16 GB consumer with grad accumulation) and about 15 GB of data for
200 h; reference dataset HiFiTTS-2 (NVIDIA-licensed) with Kyutai-published alignments **[D]**.
Evaluation itself requires CUDA **[C]**.

### 2.5 Failure modes they documented (all of them real for us)

End-of-sentence repeats (fixed in the `_2` models, issue #347), stutters, digits read once too often,
start-of-clip bursts/clicks (a 5 ms fade-in patches a Mimi first-sample click, `tts_model.py:514-517`),
pacing, mis-pronounced last word when terminal punctuation is missing, EOS never firing or firing
mid-sentence (tuned by `eos_threshold`), and quality of a clone mirroring the prompt **[D]/[C]**.
Cannot insert silence or pauses via text **[D]** `README.md:238-240`. Not thread-safe; batch size fixed
at 1; `quantize=True` is CPU-only; `torch.set_num_threads(1)` is the recommended single-stream setting;
torch 2.4.0 produces wrong audio **[C]**.

---

## 3. Licensing and provenance — the part both projects did worst

OpenWhispr is MIT, which licenses *its code* only. Its runtime pulls assets whose licences it does not
record **[C]**:

- `nircmd.exe` from nirsoft.net, **"Free for non-commercial use"**, shipped in packaged Windows builds
  while `README.md:119` states the product is free for personal *and commercial* use — a direct
  contradiction, and the single hardest flag found in either archive.
- ONNX Runtime from `github.com/csukuangfj/onnxruntime-libs`, a **third-party mirror**, not Microsoft.
- Whisper GGML weights from their own fork tag `0.0.10`; **size-checked, not hash-verified**.
- Orukeet weights `CC-BY-SA-4.0` (**share-alike**) — the only licence field present in the entire
  model registry, from the same vendor family that operates the cloud endpoint.
- Silero VAD, pyannote segmentation, 3D-Speaker CAM++, MiniLM embeddings: downloaded from mirrors with
  **no licence recorded**.
- A deliberately positive finding: they switched to sherpa-onnx `-no-tts` archives because the TTS
  builds statically link espeak-ng under **GPL-3.0** — evidence that a copyleft audit happened here and
  changed a build decision **[C]** `CLAUDE.md:255`.

Pocket TTS: code MIT (no copyright line); **model weights' licence is not stated in the repository**, the
voice-cloning checkpoint set is gated behind accepting terms, and an ungated without-cloning mirror is
used as an automatic fallback that silently disables cloning **[C]** `tts_model.py:259-265`. The 26 preset
voices are attributed to VCTK, EXPRESSO, EARS, Mozilla Common Voice, VoiceZero, "voice donations" and one
developer recording **[C]** `utils/utils.py:19-47`, with per-voice consent said to live on the Hugging Face
page — **not disclosed in the repo**. Training data is HiFiTTS-2, NVIDIA research-licensed.

**Consequence for this project:** nothing in either archive can be shipped as an asset on the strength of what
is written in it. Provenance review is a gate with teeth, and we implement it in code (§5 of
`docs/architecture.md`), not in prose.

---

## 4. Architectural lessons we are adopting, and what we are doing differently

Adopting, as concepts (no code transplanted):

1. **Raw capture + first-party gating.** Disable OS AGC/NS/AEC, measure in our own code, and treat
   "obviously good" DSP as disabled until measured — AEC3 default-off is the precedent.
2. **Lossless shadow PCM beside the codec stream**, with self-vetoing coverage checks so a short tap
   degrades to the safe path rather than producing silently-truncated audio.
3. **Partial / final / retract** as the transcription event triple, reduced by a pure function.
   Any streaming design without a retraction path will contradict itself the first time a model revises.
4. **Anti-hallucination thresholds tuned against a corpus, with the cost of the tuning recorded**
   (their 2.25 %→0.06 % figure and the reason it is skipped for meeting chunks).
5. **Continuity guard on capture**: silence for longer than a deadline is a *device failure*, not silence.
6. **Voice = exportable state, versioned**, not an opaque embedding — the Pocket TTS KV-cache snapshot is
   a good shape for a voice *file format*.
7. **Cost-asymmetric LID routing**: a false rejection costs one extra pass; a false acceptance pastes
   unrelated words into the user's conversation. Thresholds must be set in that direction of harm.
8. **Depth distillation + selective int8 quantization** as the CPU-viability levers, both with measured
   quality deltas rather than assertions.

Doing differently, because these are the reference projects' actual gaps:

- **Capability is per (task, language, engine, status), and status distinguishes measured from vendor-declared.**
  A boolean `supported` flag is what allowed OpenWhispr to carry no Bengali entry while shipping a
  61-language picker, and allowed a "pocket TTS" README to imply generality across 7 languages.
- **Approval is enforced at load time, not documented in a README.** Default-deny, digest-verified,
  approver-named.
- **Offline is a testable property**, proved by a socket-level guard plus a static import scan, rather
  than an allowlist guide for the user's firewall.
- **Thresholds come from measurement.** Profile ceilings and any claimed latency are labelled
  provisional until a benchmark run on this machine produces numbers; `measured: bool` is part of the data.
- **Benchmark receipts carry their own limits**, learned from their JSON artefacts — and unlike them, we
  record *local* RTF, because theirs contains no local model measurement at all.
- **Missing models are reported as missing.** Where this project has no approved asset, the interface exists,
  the requirement is itemised, and the call raises rather than returning silence or a tone.

## 5. Unresolved questions

See `DECISIONS.md` — the open items are approval and asset-provisioning decisions that only the user can
make, plus one environmental constraint recorded in `docs/baseline.md`: free memory at inspection time was
0.56 GB of 15.71 GB, which would invalidate any latency or RTF measurement taken in that state.
