# ASR architecture — proposed Conformer–TDT student, and what is not built

**Status: proposal.** This document records the recognition architecture the owner has chosen and the tensor
contract the new frontend implements. It is not a description of a working model. Per the repository's own tier
vocabulary (`README.md` §2, `docs/architecture.md` §7): the frontend is **BUILT and tested**; the student, its
trainer, its checkpoint, its teacher path, its tokenizer and its decoder are **PROPOSED — nothing exists**. No
number in this file is a measured accuracy, and none may be quoted as one.

Scope of this phase (Phase 1) was: a dependency-free log-mel frontend, its tests, this document, and one docstring
correction.

**Phase 1B added parity work on top of Phase 1**, in the same files plus
`tests/test_audio_features_parity.py`: a named teacher-parity configuration
(`FBankConfig.whisper_parity()`), the general DFT that `n_fft=400` requires, Slaney mel with continuous edge
interpolation and area normalisation, reflect centering with the reference's 160-sample tail pad, the
whole-buffer dynamic floor and rescale, and the executable comparison in §3.3. The student default geometry was
left alone: Phase 1B added a second contract rather than silently rewriting the first. Nothing else was changed. The Conformer–TDT adapter, the trainer, the corpus-batch scorer, the
provisioning allowlist, the approval/licence rules, the teacher-target gates and every checkpoint format remain
unbuilt and untouched.

---

## 1. What exists now, with paths

| Component | Path | State |
|---|---|---|
| Log-mel frontend | `src/nativevoice/audio/features.py` | BUILT and arithmetic-verified. Two named contracts: the **student** default geometry, and `FBankConfig.whisper_parity()`, which matches the installed reference extractor to ≤1.0e-05 (§3.3). Neither is implied by the other |
| Feature contract tests | `tests/test_audio_features.py` | BUILT |
| Canonical audio | `src/nativevoice/core/audio.py` | pre-existing, unchanged except its docstring (§6) |
| int16 PCM maths | `src/nativevoice/audio/pcm.py` | pre-existing, reused (`FULL_SCALE`) |
| Resampling | `src/nativevoice/audio/resample.py` | pre-existing; the frontend **refuses** to call it for you |
| Recognition contracts | `src/nativevoice/stt/api.py` | pre-existing; **no student engine implements `TranscriptionEngine`** |
| Engine catalogue + routing | `src/nativevoice/stt/engine_registry.py`, `language/routing.py` | pre-existing, capability-led |
| Only real STT engine today | `src/nativevoice/stt/adapters/faster_whisper_engine.py` | pre-existing third-party CT2 adapter |

## 2. The feature contract as built

**Status: two contracts, deliberately not merged.** The default `FBankConfig()` is the project's own student
frontend — arithmetic-verified, reproducible, and **not** teacher-compatible: §3.2/§3.3 measure that gap at
15–18 % mean of the reference's usable span. The named alternative `FBankConfig.whisper_parity()` reproduces the installed
reference extractor to ≤1.0e-05 (§3.3). Both exist so that which one a student trains on is a recorded decision
(D1) rather than an accident of whichever geometry the code happened to reach first. A green parity test is not
an endorsement of either choice, and no Phase 2 consumer may treat one config's results as the other's.

A feature grid is the only interface Phase 1 defines. Everything downstream consumes it unchanged or says so.

    FBankConfig defaults:  sample_rate 16000 Hz
                           window_ms   25.0  -> window_samples 400
                           hop_ms      10.0  -> hop_samples    160
                           fft_size    1024  -> bin_count      513   (zero-padded DFT, unnormalised forward)
                           n_mels      80
                           f_min       20.0 Hz,  f_max  Nyquist (8000 Hz)
                           log_floor   1e-10  (a floor in energy units, applied BEFORE the log)
                           window      "povey"  (also "hamming", "hann")
                           power       "power"  (also "magnitude")

    log_fbank(AudioBuffer, FBankConfig) -> FeatureGrid
      FeatureGrid.frames            int
      FeatureGrid.values            array('d'), row-major, len == frames * n_mels
      FeatureGrid.value(f, m)       float        FeatureGrid.row(f) -> tuple
      FeatureGrid.extent_s(f)       (start_s, end_s) of the samples that frame came from

  * **Input:** interleaved int16 `AudioBuffer`. Stereo is downmixed by `AudioBuffer.to_mono()` (averaging, the
    same rule `stt/orchestrator.py:129-131` applies before an engine). A buffer whose rate differs from
    `config.sample_rate` raises `ConfigurationError` naming `nativevoice.audio.resample` — the frontend never
    resamples silently, because a model trained on one rate and fed another is not detectable downstream.
  * **Frames:** no padding at either end, which has a cost that is now tested rather than assumed: with a
   hop longer than the window, samples that fall between two windows are observed by *nothing*, and a burst
   placed there leaves the grid byte-identical (`test_samples_between_sparse_frames_are_simply_not_observed`,
   with the positive control that the same burst inside a frame does change it). `frames = 0 if n < window else 1 + (n - window) // hop`. One second at
    16 kHz is exactly **98** frames; 399 samples is 0, 400 is 1, 559 is 1, 560 is 2 (all asserted).
  * **Scaling:** samples ÷ 32768.0 (`pcm.FULL_SCALE`); unnormalised DFT so Parseval holds as
    `Σ|X|² = fft_size · Σ|x|²` over the two-sided spectrum (asserted to 1e-12 relative).
  * **Mel scale:** HTK, `2595·log10(1 + f/700)`, `n_mels + 2` points spaced uniformly in mel, converted back to
    Hz and snapped to FFT bins with explicit round-half-up (`math.floor(x + 0.5)`, not Python's banker's
    rounding) so the mapping is reproducible outside Python.
  * **Filters:** triangular in frequency, **no area normalisation**, stored sparse (`first_bin`, `weights`), peak
    weight exactly 1.0, endpoints exactly 0.0, `f_min = 20 Hz` keeps bin 0 (DC) out of every filter.
  * **Log:** natural log of `max(energy, log_floor)`. A silent frame is therefore exactly
    `-23.025850929940457` in every bin, never `-inf` (asserted on `fixtures.silence()`).
  * **Precision:** computation and storage are float64 (`array('d')`). **No float32 cast happens here.** Casting
    is a consumer decision at the model boundary and is where roughly 7 decimal digits of rounding enter; hiding
    it inside the frontend would make two "identical" pipelines non-comparable.
  * **Determinism:** no RNG, no file access, no environment reads, no mutable module state, nothing cached. Same
    input twice → identical values, different arrays (asserted).
  * **Amplitude linearity:** doubling the integer samples raises every bin by exactly `ln 4` (asserted to 1e-9),
    which is the property that makes the absence of normalisation auditable.

### 2.1 Deliberately absent from the frontend, and why

Not implemented, not hidden, not defaulted:

* **pre-emphasis** — a 0.97 pole is a Kaldi convention that changes spectral tilt. A knob nobody has verified
  against a reference is worse than no knob. Add it with a measurement, in a later phase.
* **CMN / lnorm / spec_augment / LF-MF / delta-delta** — training-recipe decisions. `docs/provenance.md` and
  `DECISIONS.md` record that no recipe or corpus is approved, so a normalisation choice here would be a decision
  made by the person who happened to type it.
* **float32 conversion, batching, padding-to-batch, and any torch/numpy tensor object** — see §5.
* **VAD/endpointing interaction** — segmentation stays in `audio/vad.py` and `stt/orchestrator.py`; the frontend
  is per-buffer and stateless, so it cannot silently disagree with the segmenter.

## 3. Reference parity — §3.2 the student geometry fails it, §3.3 the parity config meets it

Verified, by tests that can fail:
* the radix-2 FFT equals an independently written naive O(N²) DFT on a fixed 64-point sequence (relative error
  < 1e-12), and equals `numpy.fft.rfft` when numpy is importable (skipped, labelled NOT RUN, when it is not);
* Parseval over the two-sided spectrum;
* `power="magnitude"` is the elementwise square root of `power="power"` (exact);
* every grid row equals a from-spectrum-and-filters recomputation to 1e-12 nats;
* closed-form spectra: an impulse is `|X[k]|² = 1` in every bin; an all-zero frame is exactly 0.0; a DC frame is
  `N²` in bin 0 and nothing elsewhere; a cosine centred exactly on bin 5 peaks in bin 5 with power `N²/4`
  and `|X| = N/2`, neighbours 0;
* geometry, frame counts (399/400/559/560/719/720 boundaries), floor behaviour, window endpoints, monotonic tone
  response, contiguous bin coverage with DC and Nyquist excluded;
* the error introduced by snapping mel edges to FFT bins is **bounded and measured**, not waved through:
  tone frequencies recovered from the grid land within 12 Hz at 200 Hz, 25 Hz at 1 kHz and 60 Hz at 5 kHz;
* the grid-shape invariant is enforced (`row()` can no longer return a silently short tuple), and the default
  geometry's sample-rate limit (works at 8/16/22.05/24 kHz, refused at 32/44.1/48 kHz unless `fft_size` grows)
  is pinned by tests.

**NOT verified:** parity with any external *speech* frontend — Kaldi `fbank`, `torchaudio`, librosa
`feature.mel`. None of those is installed, installing one was out of scope, and the network was not used.

Two things **were** established about the intended teacher's own extractor. First, by reading the source of the
already-installed `faster_whisper` package (`feature_extractor.py`, a local read-only file inspection under
`site-packages/faster_whisper/`), because that package is what the planned teacher
actually runs — the two frontends differ on nine axes, and the differences are line-referenced rather than
remembered:

### 3.1 Axis-by-axis differences, from the reference source

| Aspect | this frontend (`features.py`) | installed `faster_whisper` frontend |
|---|---|---|
| transform length | `fft_size=1024` for a 400-sample window (zero-padded, interpolated bins) | `n_fft=400` (= the window; no padding) `:11` |
| hop | 160 samples (10 ms) | `hop_length=160` `:9` — **the one agreement** |
| window | Povey `(0.5−0.5cos)^0.1`, endpoints exactly 0 | periodic Hanning, `np.hanning(n_fft+1)[:-1]`, float32 `:213` |
| framing | no padding: `1 + (n−400)//160` | **center** padding `pad_amount = n_fft // 2` `:118` → different frame count and offsets |
| mel scale | HTK `2595·log10(1+f/700)` | Slaney: linear below 1000 Hz, log above, `mels ∈ [0, 45.2456]` `:33-50` |
| low-frequency limit | `f_min = 20 Hz`, so DC bin is in no filter `:1` | mel range starts at **0 Hz**, so DC is included `:33` |
| filter area | no normalisation | Slaney area renormalisation `enorm = 2/(hi−lo)` `:62-63` |
| spectral measure | `power` (\|X\|²) default, `magnitude` selectable | `\|STFT\|²` `:222` |
| log | natural log of `max(energy, 1e-10)` | `np.log10(clip(mel, 1e-10))`, then `max(log, max−8.0)`, then `(log+4)/4` `:226-228` |
| precision | float64 end to end | float32 |

### 3.2 Measured on identical input, under the student geometry

Second, and more usefully than a table of source differences: the teacher's extractor was **run on the identical
buffer** as `log_fbank`, with only already-present packages (`faster_whisper.feature_extractor.FeatureExtractor`,
`feature_size=80`, `n_fft=400`, `hop_length=160`, `sampling_rate=16000`, `chunk_length=30`, driven through
`numpy`). This was done in a throwaway script **outside** the repository — no install, no network, nothing added
to `src/`, `tools/` or the test suite — and my natural-log grid was converted into the teacher's exact
post-processing (`log10` → clip at −10 → `max(x, max−8)` → `(x+4)/4`) so the numbers are in the teacher's own units:

| Input, 1 s | teacher frames | my frames | mean \|diff\| | max \|diff\| | teacher range | my range |
|---|---|---|---|---|---|---|
| 1 kHz tone | 101 | 98 | **1.413** | 2.382 | −0.67 … 1.33 | 0.54 … 1.94 |
| 300→3500 Hz sweep | 101 | 98 | **1.231** | 2.591 | −0.83 … 1.17 | 0.14 … 1.79 |

Shifting my frames by 1 and 2 hops does not reduce the difference (1.413 → 1.424 → 1.431), so this is **not** a
framing-offset artifact — it is the values. The teacher clamps its own useful span to 8 units (the `max − 8.0`
rule), which makes a mean absolute difference of ~1.2–1.4 roughly **15–18 % of the entire dynamic range the model
ever sees**, with my grid also sitting about 1.2 units high at the same time.

Consequences, stated as conclusions rather than caveats:

1. **Parity with the selected teacher is not achieved, and the size of the gap is now measured, not guessed.**
   Nothing in this repository claims equivalence.
2. A student trained on these features and distilled against Whisper large-v3 would be comparing its view of the
   audio against a teacher's view of materially different numbers. Until D1 below is ruled, the §2 input contract
   is **PROVISIONAL** — provisional for training use, not provisional for the arithmetic it performs.
3. Whisper's last three lines are a **normalisation the teacher bakes into its input** (`max−8` floor, `(x+4)/4`);
   this module applies none, by design (§2.1). So "add CMN later" is not the only gap: matching or deliberately
   not matching that exact transform is a decision that has to be made and recorded.
4. No accuracy figure from any model consuming these features may be attributed to "the standard frontend".
   Reproducing §3.2 requires re-running that external extractor, whose parameters are quoted above but whose
   behaviour is not pinned by any test in this suite — a change to the installed package would silently move the
   reference.

### 3.3 Parity achieved against the reference, under a named configuration

Re-running the same comparison after adding `FBankConfig.whisper_parity()`: the identical float32 input derived
from the same int16 buffer, the reference's own `FeatureExtractor(feature_size=80)` from **faster-whisper
1.2.1** with numpy 2.5.3, and this frontend computing in float64.

One method for both columns: the student grid is mapped into the reference's units (the same
`log10 → clip → max−8 → (x+4)/4` chain), then compared; the parity grid is compared raw, because it already
produces those units.

| Input, 16 kHz | student config, in teacher units | frames | **parity config, raw** | frames |
|---|---|---|---|---|
| 1 kHz tone, 1 s | mean 1.413, max 2.382 | 98 vs 101 | **mean 1.22e-08, max 9.47e-07** | 101 ✓ |
| 300→3500 Hz sweep, 1 s | mean 1.231, max 2.591 | 98 vs 101 | **mean 1.04e-07, max 1.00e-05** | 101 ✓ |
| 1010 Hz off-centre, 0.5 s | mean 1.312, max 1.667 | 48 vs 51 | **mean 2.13e-07, max 5.32e-06** | 51 ✓ |

Frame counts and shapes now agree on every case tried — 101 per second, 51 for 0.5 s, 26 for 4000 samples,
3 at the 400-sample boundary, 2 at 160 samples, all `(80, frames)` — and the worst case over eleven inputs is
**1.0e-05** on an 8-unit usable span — about **1.3e-06 %** of the range the teacher sees, against mean
**15–18 %** and peak **21–32 %** for the student geometry in the same units.

Why the residual is precision rather than a different formula: the reference computes in float32 (window cast,
complex64 STFT, float32 mel matrix) and this frontend in float64. Restating the reference's *own* algorithm in
float64 and comparing that to this implementation agrees to **≤4.8e-09**, while the float32 reference against the
same float64 restatement differs by ~1e-05. Agreement at 5e-09 with a shared formula, and disagreement only at
float32 scale, is parity; two similar-but-different frontends would not close by three orders of magnitude when
only precision changed.

Standing conditions, all three load-bearing:

* **Environment-dependent, and the skip was demonstrated rather than asserted.** With `faster_whisper` made
  genuinely unimportable (a `MetaPathFinder` raising `ModuleNotFoundError`), the 21 reference-dependent tests report
  `NOT RUN: the reference feature extractor (faster-whisper) is not installed on this machine` and **111 still
  pass**; with `numpy` blocked as well, 22 skip and 110 pass. Note what that experiment also revealed: a module
  that exists but fails to import is reported as a **failure**, not a skip, so a broken install cannot masquerade
  as an absent one and quietly drop parity out of the suite. The dependency-free `TestParityContract` pins only
  what can be pinned without the package: the geometry field by field, the Slaney constant, the frame arithmetic
  and the reflect-pad order.
* **The reference is not pinned by any lockfile or CI** (the repository has no `.github/`). Its identity is
  recorded above — faster-whisper 1.2.1, `n_fft=400`, `hop_length=160`, `feature_size=80`, `center=True` with
  `mode="reflect"`, 160-sample tail pad, unnormalised `rfft`, `|STFT|²` on `[..., :-1]`, Slaney filters scaled by
  `2/(hi-lo)`, `log10(clip(x, 1e-10))`, `max(x, max−8)`, `(x+4)/4`. An upgrade can move the target and only the
  test would notice.
* **Parity mode uses the general DFT**, because 400 is not a power of two: 0.60 s per second of audio on one core,
  against 0.16 s for the student radix-2 path. Neither is a streaming geometry; neither is recorded in
  `artifacts/benchmarks/`.

Two reference behaviours were adopted only because the reference has them, and both are flagged rather than
praised: reflect centering with a tail pad, and the **whole-buffer** `max(x, max−8)` clamp followed by `(x+4)/4`
— the second makes one frame's values depend on the loudest frame in the same call, so a parity-mode grid is not
frame-reproducible in a stream. `FBankConfig.global_postprocessing` reports which contract produced a grid, and
the student default keeps it `False`.

### 3.4 What the final review added, and what it corrected about §3

The checkpoint review re-inspected the implementation rather than the report, and it changed three things:

* **A fabricated limitation was removed.** The reflect pad once refused any buffer shorter than the pad,
  on the belief that `numpy.pad(mode="reflect")` cannot do it. It can, at any pad length — it keeps bouncing.
  `_reflected` now reproduces the triangular-bounce rule and matches numpy on **36/36** combinations of
  buffer length (1, 2, 3, 5, 40, 41, 200, 201, 999) against pad length (8, 32, 200, 512), and parity grids
  now agree down to a **one-sample** buffer (max diff 0.0 at n=1, ≤1e-07 for n=5…41, ≤5.8e-06 at n=4000).
  The earlier refusal was not conservative caution; it was a divergence from the reference dressed as one.
* **The tail pad is a parameter, and it is tested as one.** `FeatureExtractor.__call__(padding=160)` is the
  160-sample tail; a config with `tail_pad_samples=0` is compared against `padding=0` and matches to
  5.7e-07 with the expected 25-vs-26 frame count, so parity is not an artifact of copying one constant.
* **`chunk_length` is inert for this method, and now pinned.** It only assigns `n_samples`/`nb_max_frames`,
  which `__call__` never reads; a test asserts byte-equal output with and without it, and states that if a
  future version starts using them the test must fail and §3.3 must be re-measured. An earlier draft of this
  document called that path "untested and uncertain"; it is now tested.

Also corrected during review, without changing behaviour: the bin-frequency tuple is built once per filterbank
instead of once per filter; `post_scale` rejects a non-tuple as a `ConfigurationError` rather than a
`TypeError`; an empty grid now carries its source length like every other grid; and the `FBankConfig`
docstring no longer implies the collapse guard applies to both edge modes.

## 4. Proposed Conformer–TDT data flow — PROPOSED, none of it built

```
AudioBuffer(int16)                                  [BUILT]
  → resample/mono to 16 kHz by the orchestrator      [BUILT, unchanged]
  → log_fbank → FeatureGrid(frames × 80, float64)    [BUILT; contract PROVISIONAL for training — §3]
  → (proposed) cast to float32, optional CMN          [NOT BUILT]
  → (proposed) subsampling front-end (stride 2)       [NOT BUILT]
  → (proposed) Conformer blocks × N                   [NOT BUILT]
  → (proposed) TDT output layer + emission calendar   [NOT BUILT]
  → (proposed) tokens → text via a tokenizer/vocab    [NOT BUILT: no vocab exists here]
  → Recognition / TranscriptionResult                 [BUILT contracts]
```

The contracts a student **must** satisfy to join the tree, all pre-existing and unmodified
(`stt/api.py:61-93`, verified):

    engine_id, model_id, format = AudioFormat(16000, 1), supports_streaming
    capabilities() -> tuple[Capability, ...]     # per language, per Task.STT, with a status
    warm() / transcribe(request) / stream(chunks, language, token) / close()

Rules the routing layer enforces, which the student cannot negotiate:

* A capability with status `DECLARED` is not selectable when `strict=True`
  (`stt/engine_registry.py:79-94`, `core/language.py:124-125`). A fresh student therefore routes **nothing**
  until it is measured on this machine — that is the intended, honest state, not a defect.
* `record_measured` refuses a status other than `MEASURED` or a missing metric
  (`stt/adapters/faster_whisper_engine.py:296-303`, pattern to reuse).
* Scope is checked before approval, and an approval override cannot widen scope
  (`models/approval.py`, `tests/test_stt_real.py:165-183`). A research-scoped student manifest cannot reach the
  product path, by test.
* `hint_terms`/`context` are folded into one ≤900-character prompt before the engine call
  (`stt/orchestrator.py:185-192`, `stt/text_context.py:15-18`). A TDT decoder has no prompt channel, so the
  student must either implement context biasing or report that capability as `UNSUPPORTED` — never accept the
  field and ignore it.
* Empty output raises rather than returning `""` (`stt/orchestrator.py:198-203`).

Open tensor questions for Phase 2, listed as questions rather than answered: batch shape and padding convention;
whether `FeatureGrid` is passed as-is or converted to a tensor at the adapter edge; whether subsampling lives
inside the model or in the frontend; VAD-segment length limits versus `max_duration_s = 30.0`; the token
vocabulary and its provenance (none recorded anywhere in this tree).

## 5. Environment and dependency facts

* `pyproject.toml`: `dependencies = []`, extras `mic`, `onnx`, `dev`; markers `device`, `slow`, `network`,
  `requires_approved_asset`; **the string `torch` appears nowhere** (verified).
* Present and importable on this machine: `ctranslate2 4.8.2`, `faster_whisper 1.2.1`, `onnxruntime 1.30.0`,
  `numpy 2.5.3`, `psutil`, `sounddevice`, `pytest 9.1.1`.
* Absent: `torch`, `transformers`, `sentencepiece`, `soundfile`, `librosa`. `DECISIONS.md` S2 records the owner's
  ruling as **no installs**, and S2(c) already reasons that CPU torch cannot train a 100M-parameter model here.
* Measured hardware context from the repository's own record: 12 cores, 15.71 GB RAM, 4 GB VRAM
  (`DECISIONS.md` S2). Training a Conformer-class student on that machine is not viable as designed `[inference
  from the recorded figures, not a measurement taken in this phase]`.
* Frontend cost, measured on this machine during development (single runs, **not** a benchmark, and not
  recorded in `artifacts/benchmarks/`): the student radix-2 path takes 1.0 s of audio in 0.16 s and
  2.0 s in 0.33 s (~650 frames/s); the **parity** path, which must use the general DFT because 400 is not a power
  of two, takes 0.60 s per second of audio (~4x slower). A 30-second utterance is therefore roughly 5 seconds
  on the student path and 18 seconds on the parity path — fine for offline work, **not** interactive or
  streaming, and no performance claim of any kind is made here.
* **Sample-rate geometry limit:** the default `fft_size=1024` works at 8000, 16000, 22050 and 24000 Hz. It is
  refused at 32000 Hz (the 80-filter ladder collapses) and at 44100/48000 Hz (a 25 ms window exceeds the
  transform). Growing `fft_size` with the rate makes 48000 Hz work (`fft_size=2048` → 1200-sample window); both
  the refusals and the working rates are pinned by tests, so this cannot drift into a silent surprise.
* **Parity is defined at 16 kHz only.** It reproduces a reference geometry of `n_fft=400` at `sampling_rate=16000`
  with a hardcoded mel top of 45.2456 mel. `whisper_parity()` accepts no rate argument on purpose: reproducing the
  reference at another rate would need that frontend's own numbers there, which nothing in this repository records,
  so approximating them by rescaling `fft_size` would be a claim, not a parity result.

## 6. The docstring correction made in this phase

`core/audio.py` claimed "Float is available at the boundary (`to_float32`)". No such method exists on
`AudioBuffer`; the real conversion is `audio.pcm.to_float_samples` (an `array('d')` scaled by 32768.0), plus
adapter-local conversions (`faster_whisper_engine._to_float32` for numpy float32). The docstring now names the
functions that exist. Docstring text only — no behaviour changed, no signature added or removed.

## 7. Unresolved decisions, blocking Phase 2 and all training

These are the owner's to rule on. Nothing here was assumed, and none of them is a bug I can fix by typing.

| # | Decision | Why it blocks |
|---|---|---|
| D1 | **Awaiting the owner's approval.** Train on the project's own geometry (`FBankConfig()`) or on `FBankConfig.whisper_parity()`? | Both are implemented and tested; §3.2 and §3.3 give the measured trade-off. Neither has been made the default of the other. Recommendation below; it is not applied. |
| D2 | Licence, `source` and `redistributable` for a **self-trained** checkpoint | `models/api.py:96-104` and `approval.py:43-53` require non-empty `licence_spdx` and non-`None` `redistributable`; `docs/audit-register.md` Addendum 27 and `DECISIONS.md` N11 record that the repo still carries no `LICENSE`, `AUTHORS` or ownership record. |
| D3 | `*.pt` / `*.safetensors` / `*.yaml` checkpoint formats | `tools/provision.py` `classify_asset` refuses them today; touching that allowlist is a licence-facing change, explicitly out of Phase 1 scope. |
| D4 | Whether the student is `MEASUREMENT`-scoped first | The existing fence proves a measurement manifest cannot reach the product path — which is correct, and also means a student cannot "just work" in the product. |
| D5 | Trainer vs. the working-tree fences | `tests/test_corpus_reader_fence.py` requires any training-shaped module to name `corpus_training_blockers`/`corpus_training_eligible`, and `CORPUS_MANIFEST_READERS` is allowlisted to `tools/corpus_forge.py` alone. `GATED_TRAINING_CALLERS` is empty because no trainer exists. |
| D6 | Teacher targets and pseudo-labels | `tools/corpus_forge.py:183-200` refuses until the large-v3 `NOASSERTION`/`redistributable=False` conflict and source-data rights both close. Unchanged in this phase. |
| D7 | Training corpus with usable rights | `DECISIONS.md` S5 still says none is licensed; `artifacts/corpus/en-synthetic-01` is quarantined and synthetic audio may never enter a quality claim. |
| D8 | **EmbeddingGemma 2 scope — UNRESOLVED** | `docs/architecture.md:202-204`, `README.md` and `docs/audit-register.md` Addendum 27 record it as the **synthesis** training-side encoder, with no manifest, no provenance row and never examined for licence or availability. This phase's instruction framed it as ASR-side. I have not picked a reading; the conflict stands as recorded. |
| D9 | AI4Bharat candidate models | Zero occurrences anywhere in this repository (verified by search). Nothing has been reviewed, downloaded, named in a manifest, or licence-checked, and I assert no licence for any of them. |
| D10 | `Task` has no EMBEDDING member | `core/language.py:23-27` is closed (`stt`, `tts`, `lid`, `voice_clone`); an embedding asset has no task to register under. |


### 7.1 Recommendation on D1 (proposed, not applied)

The evidence points to **keeping the student geometry as the project's frontend and treating
`whisper_parity()` strictly as a measurement instrument** — but it is a ruling, not a fact, and it is yours.

* **Frame independence is an architectural property, not a preference.** The teacher's `max(x, x.max() − 8)`
  clamp makes every value depend on the loudest frame in the call, so a model trained on it inherits a
  normalisation that cannot be applied to one frame at a time. This repository's whole design is an
  always-on, local, barge-in-capable pipeline (`core/cancel.py`, `audio/vad.py`, `session/`), where
  frame-by-frame reproducibility is the property being preserved. Adopting the teacher's clamp as the student
  contract would trade that away to match a preprocessing choice that has nothing to do with the student's
  architecture.
* **The teacher's frontend buys little that a student needs.** Distillation targets can come from the
  teacher's *outputs* on the waveform — its transcript — which does not require the student to consume the
  teacher's feature grid at all. Only feature-level matching (comparing mel grids directly) needs parity, and
  that is one experimental option, not a requirement.
* **Parity mode carries real costs even where it is useful:** a ~4x slower general DFT, a 16 kHz-only
  geometry, and the coupling above. It is the right tool for validating that this frontend and the reference
  compute the same thing — which is exactly what §3.3 uses it for.
* **And note the dependency:** the teacher-target use case that would most want parity is currently refused
  anyway by `tools/corpus_forge.py`'s rights gate (D6). Choosing a parity frontend now would not unlock
  pseudo-labelling, and choosing it later costs only a config field.

Suggested form of the decision, if you agree: record `FBankConfig()` as the student contract for Phase 2, keep
`whisper_parity()` exported and tested as the parity instrument, and revisit only if a feature-level
distillation experiment is actually proposed. Implementation scope is unchanged in this checkpoint: no
default was altered, and both configs remain selectable.

## 8. What Phases 1 and 1B do **not** entitle anyone to claim
## 9. Reproducing the parity environment

`faster_whisper` and `numpy` are **not declared dependencies** of this package (`pyproject.toml` stays at
`dependencies = []` with the `mic`, `onnx`, `dev` extras), and Phase 1B did not change that. On this machine they
are present because the existing CT2 measurement work installed them, which is why the reference tests execute
here and will report **NOT RUN** on a clean checkout.

To reproduce the parity result:

```bash
python -m pytest tests/test_audio_features_parity.py -o addopts="" -q     # 34 passed with the reference present
python -m pytest tests/test_audio_features_parity.py -o addopts="" -q -rs  # shows the 21 NOT RUN reasons without it
```

Reference identity used for every number in §3: `faster-whisper 1.2.1`, file
`faster_whisper/feature_extractor.py` (230 lines, sha256
`e403966dbc592a53695eea2aea24fa60bab50ef6755e0076b311f907be7a397c`), `numpy 2.5.3`. If that digest changes, the
parity figures in this document are stale until re-measured.

**Recommendation, not applied (needs approval):** declare the reference as an opt-in extra —
`[project.optional-dependencies] parity = ["faster-whisper==1.2.1", "numpy"]` — so the parity tests are runnable
from a clean environment and the pin is recorded where a reviewer will find it. That is a dependency change and
an unpinned-third-party decision, both of which this phase's boundaries forbid without your say-so, so nothing was
added.


* That recognition works, improved, was measured, or is competitive with anything. No model ran, no accuracy
  number exists anywhere in this checkpoint, and a matching frontend is a precondition of a comparison, not one.
  `tests/test_audio_features.py` are tones, DC and envelopes — `tests/fixtures.py` states in its own docstring
  that they "must never be used to claim recognition or synthesis quality".
* That the **student** geometry matches a third-party frontend. It does not, by measurement (§3.2). The parity
  config does (§3.3), but only under `tests/test_audio_features_parity.py::TestAgainstReference`, which reports
  **NOT RUN** without `faster_whisper` installed — so on a machine without that package, no executed parity claim
  exists at all, and the reference itself is unpinned by any lockfile or CI.
* That `frozen=True` on `FeatureGrid` makes the feature buffer immutable. It does not: `values` is a mutable
  `array('d')` by design, the same convention as `AudioBuffer.samples`, and a test pins that fact. What the
  review added is the shape invariant, so a malformed grid cannot silently yield a short row.
* That a Conformer–TDT student, its trainer, its tokenizer or its checkpoint exists. None does.
* That any capability status was raised. Every language row in `docs/capability-matrix.md` remains as generated
  on 2026-10-12; `CapabilityStatus` was not imported by the frontend and no status was written.
