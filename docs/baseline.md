# Environment baseline — measured 2026-10-12

Every number here was produced on this machine at inspection time by the command named under it. Nothing
is copied from a spec sheet.

## Compute and memory

| Item | Value | How measured |
|---|---|---|
| OS | Windows 11, build 10.0.26200.9457, AMD64 | `cmd /c ver`, `platform.platform()` |
| CPU | 12 physical / 16 logical cores | `psutil.cpu_count(logical=False/True)` |
| RAM total | 15.71 GiB | `psutil.virtual_memory().total` |
| RAM available **at inspection** | **0.56 GiB — 96.5 % in use** | `psutil.virtual_memory().available` |
| Disk C: | 340.13 GiB free of 498.88 GiB | `shutil.disk_usage` |
| GPU | NVIDIA GeForce RTX 3050 Laptop, **4096 MiB**, driver 617.42 | `nvidia-smi --query-gpu=...` |
| Display | single 1920x1080 @ 96 DPI (100 % scaling) | verified 2026-10-12 |

The 0.56 GiB figure is the binding constraint, not the totals. Any latency, real-time-factor or peak-memory
measurement taken while the machine is in that state is invalid, so profile thresholds are seeded as
provisional with `measured: false` and the benchmark harness is built to be re-run when the machine is idle.
See `DECISIONS.md` S5.

## Toolchain

| Tool | Version | Status |
|---|---|---|
| Python | 3.14.7 | two installs on PATH (`WindowsApps` shim and `AppData\Local\Python\bin`) |
| pytest | 9.1.1 | available |
| git | 2.55.0.windows.5 | available; **no repository created yet** |
| ruff, mypy | — | **not installed** (S2) |
| torch | — | **not installed** (S2, S1c) |
| soundfile | — | not installed |
| `audioop` | — | **removed from the standard library in 3.13** — the PCM maths in `audio/pcm.py` exists because of this |

Present and usable: numpy 2.5.3, onnxruntime 1.30.0 (**CPU only** — no CUDA execution provider),
faster-whisper 1.2.1, sherpa_onnx 1.13.8, sounddevice 0.5.6, psutil 7.2.2, fastapi 0.141.1.
`socket` connectivity to `huggingface.co:443` succeeded at inspection: the network is reachable, so
offline behaviour must be *enforced by test*, not assumed from a cable being unplugged.

## Speech assets already resident

`~/.cache/huggingface` totals **5,617 MB**, all of it faster-whisper CT2 conversions:

| Model | Size on disk |
|---|---|
| Systran/faster-whisper-large-v3 | 2,947.7 MB |
| Systran/faster-whisper-medium | 1,459.7 MB |
| Systran/faster-whisper-small | 463.7 MB |
| Systran/faster-whisper-small.en | 463.6 MB |
| Systran/faster-whisper-base | 141.0 MB |
| Systran/faster-whisper-base.en | 140.9 MB |

No sherpa-onnx models are present (`~/.cache/sherpa-onnx` does not exist). `~/.cache/whisper` does not exist.
`~/.cache/pocket_tts` does not exist. This is the entire basis of the measurement options in S1 — nothing
else can be benchmarked today without downloading, which I will not do without your per-asset approval.

Note that `large-v3` at ~3 GB of weights against 4 GB of VRAM and 0.56 GB of free RAM cannot be measured in
the current state, and `base` at 141 MB can. That difference is exactly what Lite/Balanced/Performance is
for, and it will be derived from runs rather than asserted.

## Audio hardware (enumerated via sounddevice, 20 devices)

| Role | Device | Channels | Default rate |
|---|---|---|---|
| **Input (default)** | Microphone Array (Intel Smart Sound Technology for Digital Microphones) | 4 | 44,100 Hz |
| **Output (default)** | Speakers (Realtek(R) Audio) | 2 | 44,100 Hz |
| Alt input | Microphone Array 3/4 (SST) | 4 | **16,000 Hz** |
| Alt output | Speakers (Realtek) 8-ch | 8 | 48,000 Hz |
| Loopback | Stereo Mix (Realtek HD Audio Mic input) | 2 | 44,100 Hz |
| Other | Headphones 1/2, PC Speaker, Nahimic mirroring Wave Speaker | — | — |

Two consequences for the design: the canonical 16 kHz capture rate is *natively available* on some mic
endpoints (so resampling can be avoided by selecting device index 12/13), and `Stereo Mix` exists, which
makes system-audio capture testable on this machine without a virtual cable. No WASAPI loopback
process-private capture is implemented; that is a documented gap, not a silent absence, because the
reference project needed a dedicated native AEC helper for it and shipping a compiled binary is an
unapproved provenance decision.

## Accelerator matrix — measured, not assumed

| Runtime | Version | GPU usable here | Evidence |
|---|---|---|---|
| ctranslate2 (behind faster-whisper) | 4.8.2 | **yes** — reports 1 CUDA device | `ctranslate2.get_cuda_device_count() == 1` |
| onnxruntime (and therefore any sherpa-onnx ONNX model) | 1.30.0 | **no** — CPU only | `get_available_providers() == ['AzureExecutionProvider', 'CPUExecutionProvider']`, no `CUDAExecutionProvider` |
| torch | absent | n/a | not installed |

The RTX 3050 with 4 GiB is real but only *partly* supported: the same machine can accelerate a CT2 Whisper
model and cannot accelerate an ONNX Parakeet model without a different onnxruntime build. This is the
concrete case §10 warns about — "do not assume that every powerful computer has a supported GPU or that GPU
execution is always faster" — and it is why accelerator support is a per-engine fact in the hardware layer
rather than a boolean called `has_gpu`.

## Reference archives

| Archive | Extracted size | Files | Working copy |
|---|---|---|---|
| `openwhispr-main (1).zip` | 41 MB | 3,230 | local research extract, outside this repository |
| `pocket-tts-main.zip` | 2.1 MB | 166 | local research extract, outside this repository |

The original archives were left unmodified and the ZIPs were never opened for writing. The extracted copies are
read-only research references held outside the project directory, so no reference code can be imported by the
deliverable. Neither upstream is redistributed by this repository, and its licence terms were not reviewed here,
so nothing in this section clears them for redistribution.

## Workspace

`[USER_HOME]\Documents\Qoder\2026-10-12\76969c5c` was **empty** at inspection. It is the standalone
project root per §2 of the specification — separate from the main ULTRON repository, which has not been
read, referenced or modified, and which has no dependency on anything created here.
