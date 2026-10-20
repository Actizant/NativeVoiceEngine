# TTS performance — measured baseline, execution headroom, and asset candidates

**Proposal only, and superseded as a plan.** Nothing here was downloaded, installed or measured; §1 and §2
stay as evidence, §3 and §4 as history — synthesis now heads for our own flow-matching student (`architecture.md` §7).

## 1. Measured baseline: Kokoro-82M int8 multi-language v1.1

| Property | Value | Evidence class |
|---|---|---|
| RTF | **1.59 best** (8 threads); 2.01 at 1; 1.97 at 12 | measured on this machine |
| First audio, two-sentence reply | **5.06 s** at 24-token chunks; 11.55 s at 50 tokens | measured |
| First audio, single short sentence | 1.88 s | measured |
| Peak RSS | 1,001–1,046 MiB with the ASR model resident; 708.8 MiB in the evaluation run | measured |
| Load | 2.56–3.7 s per language instance | measured |
| Language switch cost | **+3.32 s** (engine rebuilt) | measured |
| Languages renderable | `en-us`, `en-gb`, `zh` — asset file list, not a model card | verified |
| Intelligibility evidence | English round-trip WER **0.0494**; Chinese CER **0.463** (failed) | measured, synthetic speech only |
| Distribution constraint | bundles **espeak-ng data under GPL-3.0** | licence text read from the package |
| License | Apache-2.0, text shipped inside the asset | verified |
| GPU use | impossible — `onnxruntime` exposes only `AzureExecutionProvider` and `CPUExecutionProvider` | measured |

**Conclusion:** CPU-only RTF above 1.0 at every setting tried, so conversational speech is not achievable
by tuning alone. Tuning is still worth taking, because it is free and it moves first-audio.

## 2. Execution headroom, no new asset

| Optimisation | Status | Expected effect |
|---|---|---|
| Chunk budget 48 → 24 tokens | **applied**, pinned by a test | first audio 11.55 s → **5.06 s**; RTF unchanged |
| Thread ceiling 8, not 12 | measured, encoded as advice | 12 threads is slower — oversubscription on 16 logical cores sharing the machine |
| Play while generating: per-chunk callback | **available, not implemented** — `OfflineTts.generate` accepts a callback | playback starts after chunk 1 instead of after the whole reply; no RTF change |
| Keep one engine per loaded language | not implemented | removes the +3.32 s language switch, at the cost of extra resident memory (to be measured, not assumed) |
| Reply shaping — short first sentence | host-side, free | shorter first chunk ⇒ earlier speech; belongs in the answer generator, not the engine |
| Stop re-resampling per request | already 24 kHz end to end | negligible |

The first two are done. The next two are code changes we can measure without approving anything, and they
improve perceived latency without pretending to fix RTF.

## 3. Candidate assets — identity, provenance, terms, and what we do not know

Expected performance below is **inference from package size, architecture and published claims. None of it
is measured on this machine.** That distinction is the whole point of the table.

| Candidate | Identity | Terms | Runs on installed runtime? | Expected vs baseline | Language coverage | Quality evidence | Verdict |
|---|---|---|---|---|---|---|---|
| **KittenTTS nano 0.8 int8** | `KittenML/kitten-tts-nano-0.8-int8`; sherpa package `kitten-nano-en-v0_8-fp32` 60.9 MB / nano v0_1 fp16 25.6 MB | **Apache-2.0** (HF tag, verified) | **yes** — `OfflineTtsModelConfig.kitten` exists in sherpa-onnx 1.13.8 | much smaller model ⇒ expect a large RTF and first-audio gain; magnitude unknown | English only | none here; upstream claims only | **was the recommended first trial; superseded by `docs/architecture.md` §7** |
| Kokoro int8 multi-lang **v1.0** | `kokoro-int8-multi-lang-v1_0.tar.bz2`, 126.2 MB | Apache-2.0 (same family) | yes | marginal (same architecture, older weights) | en, zh | current asset is v1.1 | no |
| Kokoro **fp16 / q4** ONNX | `onnx-community/Kokoro-82M-v1.0-ONNX` variants (`model_fp16.onnx`, `model_q4.onnx`) | Apache-2.0 | partially — needs a new packaging path | fp16 on CPU ONNX is usually **slower**, not faster; q4 cuts memory, accuracy unknown | en, zh | none here | not recommended |
| **Supertonic 3** | `Supertone/supertonic-3`; sherpa int8 package 122.8 MB | **OpenRail** — not an OSI-approved licence; terms must be read before any use | yes (`supertonic` config) | vendor positions it as a small, fast, on-device model | multilingual claims, unverified | none here | **blocked pending his approval on terms** |
| Matcha-icefall (en/zh) | `csukuangfj/matcha-icefall-*`, packages 73–75 MB | **no licence tag on any repository found** | yes (`matcha` config) | plausibly faster than an 82M LM | en, zh | none here | **fails our own provenance gate** — same category as the piper Indic voices already rejected |
| Piper / Coqui VITS | `vits-piper-*`, `vits-coqui-*`, 64–117 MB | no licence tag on the Hugging Face repositories | yes (`vits`) | small and fast, the classic CPU choice | many single-voice locales | none here | fails the provenance gate |
| MeloTTS-English | `myshell-ai/MeloTTS-English` | **MIT** (HF tag, verified) | **no** — needs its own Python runtime, which pulls torch | unmeasured | en (+ others upstream) | none here | only if a torch install is separately approved |

### The gap no candidate closes

Every permissively licensed option above is English-and/or-Chinese. **Bengali and Hindi synthesis is not
obtainable from this candidate set**: the Indic TTS models found earlier (`SaudxInu/indic-tts-lora-bengali-*`,
Apache-2.0; `DeepDiveDev/Bengali_finetuned_speecht5_tts`, MIT) need a Python model stack we do not have
installed, and their quality is unverified by anyone on this machine. Multilingual coverage is therefore a
separate decision from latency, and should be sequenced after it.

## 4. What was proposed at the time, in order — now superseded by `docs/architecture.md` §7

1. Implement the two no-asset changes (per-chunk callback playback; per-language engine cache) and re-run
   the latency lab. Free, reversible, and it makes the remaining gap honest rather than padded. These two
   remain open against the **current** asset, whichever synthesis model eventually answers for it.
2. Approve **KittenTTS nano int8** only (25.6–61 MB, Apache-2.0, zero new packages). Then: provision it with
   a manifest and digests, run the same round-trip evaluation, and publish its measured RTF, first audio,
   peak RAM and English WER **next to** the Kokoro numbers rather than replacing them. **Not the route now.**
3. Keep both engines selectable per voice identity. Providers stay interchangeable behind
   `SpeechSynthesizer`, so a faster English voice and a future Indic voice coexist without an architecture
   change, and a candidate that measures badly simply never gets an identity that routes to it.
4. Treat Indic and Bengali/Hindi coverage as its own proposal with its own licence review — not as a
   checkbox under this one.

## 5. Explicitly not claimed

The approved Kokoro asset stays in place until the owner approves a specific change; nothing in §3 or §4 is a
decision, and no candidate here has been adopted.
No candidate has a measured number in this document. Supertonic's OpenRail terms and the untagged
matcha/piper repositories are **not** cleared for distribution. The espeak-ng GPL-3.0 obligation applies to
every option that uses its data — which is all of them — and remains the blocking item for shipping a
distributable bundle, independent of which voice wins.
