# Decisions

## Status as of 2026-10-13

| # | Decision | Status |
|---|---|---|
| S1 | Third-party weights as a measurement instrument | **Answered** - pretrained models approved for local use with licence verification first; Kokoro (Apache-2.0) and faster-whisper (MIT) provisioned and digest-verified |
| S2 | Package installs | **Answered** - nothing installed; ASR fine-tuning **held** |
| S3 | Bengali/Hindi promises | **Answered** - infrastructure plus an honest per-language ledger; hi/bn synthesis and acoustic LID remain documented gaps |
| S4 | Project location and name | **Partly answered** - namespace renamed to `nativevoice`; the working directory is still the Qoder session folder; see N6 and N9 |
| S5 | Benchmark headroom and evaluation audio | **Partly answered** - benchmarks now run with machine context recorded beside every number; still no licensed human-speech corpus |
| S6 | Voice cloning assets | **Answered** - out of the feature set; no permissively licensed local zero-shot asset was found |
| S7 | Definition of complete | **Answered** - evidence-based status per capability, never an overall "complete" while criteria are unmet |

### Newly open

**N6 - data directory name.** The local data root is still `%LOCALAPPDATA%\ultron-voice`. Renaming it would
strand the verified assets, the model manifests and the installed voice identities, so it was left alone.
Say the word and I move it and re-verify every digest; nothing else in the code depends on the old name.

**N7 - commit author line.** The initial commit carries a `Co-Authored-By:` trailer naming an AI model. It is
local and unpushed, so removing it is one `git commit --amend` whenever you want it gone. I did not rewrite
history on my own initiative.

**N8 - TTS replacement proposal.** Real-time synthesis is not reachable with the current asset: best measured
RTF is 1.59 and first audio 5.06 s after the chunk-budget fix. The options are a different approved asset
(identity, provenance, terms, expected performance and resources presented for sign-off before any download),
or GPU execution, which needs an onnxruntime build with a CUDA provider. Nothing is downloaded until that
proposal is approved.

**N9 - repository location.** Version control now exists at the hashed Qoder session path, which is a poor
long-term home. Moving it is `git clone`/copy plus a path change; the history is two commits deep, so this
is cheap now and expensive later.

**N10 - a stray interpreter in the working tree.** `Python/pythoncore-3.15-64` (45 MB) materialised inside
the project directory during the session - not created by any command here, and deliberately **not** deleted.
It is git-ignored; removing it is yours to decide.

---

# Original decision text (2026-10-12)

Everything in `Phase B` that I can build without you is being built now. These seven items are the ones
