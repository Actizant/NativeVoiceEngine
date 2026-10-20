# KittenTTS nano trial — measured comparison against the installed Kokoro

**Scope of the approval honoured:** isolated trial only, acquired into `.trials/kitten/`, the Kokoro
installation and the shared catalogue are untouched, the default engine is unchanged, and nothing is
promoted by this document. Both engines passed the same licence gate and the same SHA-256 integrity
verification before either was loaded.

## Acquisition — verified before download, as required

| Check | Result |
|---|---|
| Repository | `KittenML/kitten-tts-nano-0.8-int8` (Hugging Face), packaged by k2-fsa |
| Bundle chosen | `kitten-nano-en-v0_8-int8.tar.bz2`, **29.8 MB** download, 43 MB installed, 360 files |
| Licence | **Apache-2.0** — front matter on the model card; the sherpa tarball publishes no licence of its own, so the upstream repo is the licence source |
| Contents actually needed | `model.int8.onnx` 24 MB, `voices.bin` 3.2 MB, `tokens.txt` 4 KB, `espeak-ng-data` 18 MB |
| Runtime | the **already installed** sherpa-onnx 1.13.8 (`OfflineTtsKittenModelConfig`) — no package was installed |
| Digests | all 360 files hashed and recorded in a trial manifest |
| Undisclosed | **training data and voice-actor consent.** The card names eight voices (Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo) and says nothing about where they came from |
| Redistribution | recorded `redistributable: False` — Apache-2.0 covers the checkpoint, not the GPL-3.0 espeak-ng data it needs, and not the undeclared voice provenance |

## Head-to-head, two independent runs

Single sentences, whole-utterance generation on both engines — the only basis on which they are
comparable, because Kitten's runtime has no chunked path. Kokoro's chunked figure is reported separately
and never mixed into the comparison. 16 trials per engine per run (8 sentences × 2 repeats), 4 and 8
inference threads, conditions recorded: 3.24 GB free, 274 processes, 16 logical CPUs.

| Metric | Kokoro int8 (incumbent) | Kitten nano int8 (candidate) |
|---|---|---|
| RTF, run at 4 threads | 2.3206 | **0.5701** |
| RTF, run at 8 threads | 2.3615 | **0.6161** |
| First audio, median | 5,888 ms / 5,991 ms | **1,994 ms / 2,156 ms** |
| First audio, best | 5,127 ms | ~1.9 s |
| Engine init | 3.64 s / 4.12 s | **1.49 s / 1.63 s** |
| Peak RSS (whole process, ASR resident too) | 1,044 / 1,049 MiB | **764 / 769 MiB** |
| Round-trip WER | 0.052 (both runs) | 0.0364 / 0.0239 |
| Round-trip CER | 0.0238 | lower, see artefacts |
| Exact matches | 10/16 both runs | 12/16 / 13/16 |
| Languages renderable | en, zh | **en only** |
| Speaker slots | 103, **unnamed** | 8, **named** |
| Offline behaviour | 0 network attempts | 0 network attempts |

Device cost, measured separately and model-independent: opening the rendering stream median **62.4 ms**
(no audio played during that measurement).

## What this does and does not establish

**Established, reproducibly:** the candidate is **the first configuration measured on this machine that
synthesises faster than real time** (RTF ≈ 0.57–0.62 where the incumbent is ≈ 2.3), with roughly **3×
faster first audio**, ~2.4 s cheaper engine init, ~280 MiB less resident memory, and no worse — on both
runs better — round-trip intelligibility on the same sentences. Threads barely move either model, which
points at a per-call ONNX cost rather than a scheduling problem.

**Not established:**
- **Quality is not judged here.** Round-trip WER measures agreement between our TTS and our ASR on
  machine-generated audio. Nothing in this document is evidence of naturalness, and nobody has listened
  to a comparison yet; the clips are reproducible from the tool.
- **Language coverage regresses.** English only, against the incumbent's English + Chinese. Neither
  provides Bengali or Hindi.
- **Distribution is not clearer.** The candidate's checkpoint is Apache-2.0, but it needs the same GPL-3.0
  espeak-ng data and discloses *no* training-data or voice-actor provenance — a stricter unknown than the
  incumbent's, which at least points at a per-voice licence page.
- **First audio is still ~2 s**, so press-to-hear remains the honest description of the experience even
  with a faster model. Real-time *generation* is not the same as conversational responsiveness.

**Recommendation as recorded at the time — now superseded, do not act on it:** adopt Kitten nano as an
*additional selectable voice*, not as a replacement — eight named voices with evidence, English, real-time
generation, lighter on RAM and load — while Kokoro remains the default and the only Chinese path. Promotion
would need its own reviewed adapter, identity packages per named voice, and its own tests, none of which
exists yet. Synthesis is now headed for this project's own flow-matching student (`docs/architecture.md` §7),
so this trial stands as a measured comparison and a licence observation, not as a route to a new dependency.

## Reproduce

```bash
python tools/provision.py --asset kitten-nano-en-v0_8-int8 --root .trials/kitten/models \
  --approved-by "Shreshtha" --authority "<trial authorisation>" --language en
python tools/tts_head_to_head.py --threads 4
python tools/tts_head_to_head.py --threads 8 --out artifacts/benchmarks/tts-head-to-head-8t.json
```

Artefacts: `artifacts/benchmarks/tts-head-to-head.json` (4 threads) and `tts-head-to-head-8t.json` (8 threads).
