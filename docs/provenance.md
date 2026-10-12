# Dependency and asset provenance inventory

Per §3 rules 7–10, code dependencies, pretrained weights, training datasets, tokenizer assets, speaker
embeddings and voice recordings are inventoried as **separate** licensing questions. This file is the record;
`models/api.py` is the enforcement.

Status vocabulary: `RESIDENT` already on this machine · `ABSENT` not installed · `APPROVED` reviewed and
cleared by a named approver · `UNREVIEWED` present but not cleared · `PROHIBITED` must not enter the product.

**A licence string below marked "per upstream" was recorded from the upstream project's own declaration and
has not been re-verified against the released artefact on this machine.** Rule 9 forbids claiming an asset is
legally safe without sufficient evidence, so those rows are `UNREVIEWED` regardless of how permissive the
licence looks, until someone reads the actual `LICENSE` of the exact version.

## 1. Python libraries the deliverable imports

| Package | Version | Where imported | Licence (per upstream) | Verification | Status |
|---|---|---|---|---|---|
| *standard library only* | 3.14.7 | `core`, `audio`, `models`, `hardware`, `config`, `language`, `stt`, `tts`, `profiles`, `session`, `eval`, `diagnostics`, `integration` | PSF / Python-2.0 | n/a | **no third-party runtime dependency** |
| `sounddevice` | 0.5.6 | lazily, only `audio/io.py` live capture+playback | MIT | not re-read | RESIDENT, OPTIONAL |
| `ctypes`/`subprocess` → `nvidia-smi` | — | `hardware/detect.py` | NVIDIA driver utility, not linked | — | RESIDENT, OPTIONAL, read-only |

Nothing in the control path imports numpy, torch, onnxruntime, psutil or fastapi. Where an optional
accelerator is used the absence must degrade gracefully, and the test suite runs with it absent.

## 2. Libraries present on this machine that the deliverable does NOT import

Recorded because their presence invites use, and each would be a provenance decision.

| Package | Version | Licence (per upstream) | Why it is not imported |
|---|---|---|---|
| numpy | 2.5.3 | BSD-3-Clause | control path stays dependency-free; PCM maths is stdlib |
| onnxruntime | 1.30.0 (CPU) | MIT | no ONNX model approved; also would imply an execution-provider story we have not measured |
| faster-whisper | 1.2.1 | MIT | candidate STT *instrument* only — see S1; would become a runtime dependency if wired to the product path |
| sherpa_onnx | 1.13.8 | Apache-2.0 | no sherpa model resident; **see the copyleft note in §5** |
| psutil | 7.2.2 | BSD-3-Clause | hardware detection falls back to `ctypes`/`os`, so it is optional |
| fastapi / uvicorn / httpx | 0.141.1 / — / — | MIT | this is a library, not a service; no HTTP surface is offered |
| torch | **ABSENT** | BSD-3-Clause | required by any ULTRON-trained synthesis model; not installed (S2) |
| soundfile | **ABSENT** | BSD-3-Clause | WAVE container is parsed directly; `audioop` is gone from 3.13+ |
| ruff, mypy | **ABSENT** | MIT / MIT | dev tooling; static checks are a hand-pass until S2 is answered |

## 3. Pretrained model assets

| Asset | Where | Bytes | Licence (per upstream) | Provenance risk | Status |
|---|---|---|---|---|---|
| Systran/faster-whisper-base | `~/.cache/huggingface` | 141.0 MB | Apache-2.0 (model card) / MIT (whisper.cpp & OpenAI code, to confirm) | converted from OpenAI weights by a third party; no digest recorded locally | **UNREVIEWED** |
| faster-whisper-base.en | same | 140.9 MB | as above | as above | UNREVIEWED |
| faster-whisper-small / small.en | same | 463.7 / 463.6 MB | as above | as above | UNREVIEWED |
| faster-whisper-medium | same | 1,459.7 MB | as above | as above | UNREVIEWED |
| faster-whisper-large-v3 | same | 2,947.7 MB | as above | cannot fit 4 GB VRAM alongside live apps; 128-lang claim is upstream's, not measured | UNREVIEWED |
| any sherpa-onnx ASR (Parakeet/Nemotron/Cohere) | **not present** | — | model licences vary; OpenWhispr records only one of them | would require download | ABSENT, not requested |
| Kyutai pocket-tts weights | **not present** | — | **not stated in the repository**; cloning set gated behind accepting terms; ungated mirror exists | gated/ungated fork changes capability silently | ABSENT, **do not fetch** |
| Pocket-TTS preset voices (26) | not present | — | per-voice consent not disclosed in repo; derived from VCTK / EXPRESSO / EARS / Mozilla Common Voice / VoiceZero / donations / one dev recording | consent basis unknown per voice | ABSENT, **PROHIBITED pending review** |
| HiFiTTS-2 (reference training set) | not present | — | NVIDIA research licence | redistribution rights unclear | ABSENT, **PROHIBITED in product** |
| Silero VAD / pyannote / 3D-Speaker / MiniLM (OpenWhispr's assets) | not present | — | **no licence recorded in their registry**; pyannote and Silero have changed upstream terms | untraceable from the archive | ABSENT, **do not adopt** |
| **ULTRON-owned weights** | to be created | — | ours | requires the training pipeline in §4 | **NONE EXIST YET** |

The honest summary: **there is not a single approved speech weight in this project.** Recognition and
synthesis quality numbers therefore cannot be produced without your S1 ruling, and the capability matrix
rightly reports `NO_MODEL` rather than an estimate.

## 4. Tokenizers, embeddings, and the training data they imply

- No tokenizer asset is shipped. TTS text processing uses our own normalisation tables and a byte/word
  budget counter; a real subword tokenizer is a training-time artefact that must be recorded separately
  (`AssetKind.TOKENIZER`) when a ULTRON model is trained.
- No speaker embedding or KV-cache voice state is shipped. `profiles/` stores user recordings plus consent
  records plus digests; the acoustic conditioning that would turn those into a voice is gated on an approved
  synthesis model and currently raises with the missing component named.
- Evaluation corpora required by §14, none present: paired audio+transcript for English (open, licence
  clearable), **Bengali and Hindi (must be approved or user-supplied)**, and cross-language identity pairs
  for cloning evaluation. `eval/samples.py` records per-sample licence and consent and refuses to let a
  synthetic fixture contribute to a quality score.

## 5. Copyleft and supply-chain notes carried forward from the reference study

- sherpa-onnx **TTS builds statically link espeak-ng (GPL-3.0)**; OpenWhispr deliberately switched to
  `-no-tts` archives for this reason. Any future TTS runtime choice must repeat that check rather than
  assume an Apache-2.0 wrapper makes the linked tree permissive.
- ONNX Runtime in the reference stack came from a **third-party GitHub mirror**, not Microsoft. Any
  binary we ever ship must name its actual origin, and a mirror is not a provenance.
- `nircmd.exe` in OpenWhispr's Windows package is **"free for non-commercial use"** while their README
  claims commercial freedom. That is the concrete shape of the risk rule 9 exists to prevent, and it is why
  the gate here is default-deny in code.
- Their Whisper GGML downloads were **size-checked, not hash-verified**; ours must record digests, and an
  unrecorded digest reports as *unverifiable*, never as verified.

## 6. Reference projects themselves

| Project | Version | Licence | Our use |
|---|---|---|---|
| OpenWhispr (Gizmo Labs) | 1.10.3 | MIT (`LICENSE:1`) | **research reference only.** No file imported, vendored, ported or renamed. Working copy lives outside the deliverable at `../_refs/`. |
| Pocket TTS (Kyutai) | 3.3.0 | MIT text present but **no copyright-holder line and no `license` field in `pyproject.toml`** | research reference only, same terms. The missing attribution is itself recorded as an ambiguity. |

MIT permits use of their *code* with attribution; §3 rules 1–4 forbid that here regardless, because the
deliverable must be original. Their architectures informed decisions documented with reasoning in
`docs/research-report.md` §4 and `docs/architecture.md` §3 — concepts, not transplants.

---

# Dated correction record - redistribution rights on installed assets (2026-10-13)

**What was wrong:** `tools/provision.py` wrote `redistributable: true` into every manifest it created, as a
hard-coded default rather than a decision. Three installed assets therefore carried a redistribution claim
that no review had established - including the Kitten trial asset, whose model card discloses no training
data and no voice-actor consent at all. That is the failure rule 9 of the mandate exists to prevent, and it
was ours, not a reference project's.

**What was done:** `tools/asset_status_audit.py` audits each manifest individually, records what evidence
exists locally and upstream, and rewrites the claim to what that evidence supports.

| Asset | Previous claim | Corrected status | Evidence inspected locally | Loads for local use after |
|---|---|---|---|---|
| `faster-whisper-base` | `true` | **unverified** (`False`) | HF tag `license:mit`; no licence text archived locally; third-party conversion of OpenAI weights; training-data provenance undisclosed | yes |
| `faster-whisper-small` | `true` | **unverified** (`False`) | as above | yes |
| `kokoro-int8-multi-lang-v1_1` | `true` | **unverified** (`False`) | Apache-2.0 text shipped inside the package, read directly; bundled espeak-ng data is GPL-3.0; per-voice consent page referenced upstream, never read or archived | yes |
| `kitten-nano-en-v0_8-int8` (trial) | never over-claimed | **unverified** (`False`) | card front matter `apache-2.0`; tarball carries no licence of its own; no training-data or voice-actor disclosure | yes (trial root) |

**Missing evidence is not the same as an established prohibition, and the record keeps them apart.** The
boolean exists only to gate loading: `None` means unknown and is deliberately refused, because a model whose
rights were never checked must not become loadable by silence. **The manifest schema has no rights-status
field**, so `unverified` cannot be stored on an asset as a value distinct from `false`: it is written as the
conservative `False`, a `provenance.permissions` sentence - "not established either way; local use approved,
redistribution claim withdrawn until reviewed" - and a dated line in `provenance.notes`, while the word
itself is carried by the machine-readable `artifacts/audit/asset-status-corrections.json`. A genuine
prohibition would be recorded there as `rights_status: prohibited` with the offending term quoted; none was
found for these four, and none was assumed.

**History preserved:** each manifest was copied verbatim to
`artifacts/audit/manifest-snapshots/*.pre-audit` before being rewritten. The manifests live in the local
data root and are not under version control, so without that snapshot the value the earlier tool wrote
would simply have disappeared.

**Still outstanding for any distributable bundle**, now stated per asset rather than in general: the
espeak-ng GPL-3.0 obligation on every option that needs its data; the Whisper conversion's upstream terms;
and a voice-consent review for any speaker asset. None of this blocks local use, which is what was approved.
