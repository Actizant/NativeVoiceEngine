# ULTRON Native Voice - round-trip intelligibility evaluation

Generated 2026-10-12. Round-trip intelligibility of the TTS+STT pair on machine-generated, licence-clean audio; NOT a human-speech quality measurement.

Self-check (en): WER=0.049383 (threshold 0.05) -> PASS

Free RAM at start: 3.846 GB. Peak RSS observed: 708.8 MiB.

| language | verdict | forced WER | forced CER | TTS RTF | TTS first-audio ms | STT RTF |
| --- | --- | --- | --- | --- | --- | --- |
| en | PASS | 0.049383 | 0.024169 | 2.3626 | 6044.4986 | 2.4427 |
| hi | UNSUPPORTED | None | None | None | None | None |
| bn | UNSUPPORTED | None | None | None | None | None |
| zh | FAIL | 1.0 | 0.462963 | 2.327 | 4510.6089 | 3.0478 |

## Per-utterance detail

### en - PASS
- ref: The quick brown fox jumps over the lazy dog.
  - forced hyp: The quick brown fox jumps over the lazy dog. (WER 0.0, CER 0.0)
- ref: She sells seashells by the seashore every summer.
  - forced hyp: She sells seashells by the seashore every sum. (WER 0.125, CER 0.073171)
- ref: Please turn off the lights before you leave the room.
  - forced hyp: Please turn off the lights before you leave the room. (WER 0.0, CER 0.0)
- ref: I would like a cup of coffee and a slice of bread.
  - forced hyp: I would like a cup of coffee and a slice of bread. (WER 0.0, CER 0.0)
- ref: The weather today is pleasant and the sky is clear.
  - forced hyp: The weather today is pleasant and the sky is clear. (WER 0.0, CER 0.0)
- ref: My favourite book is about a young engineer in the city.
  - forced hyp: My favorite book is about a young engineer in the city. (WER 0.090909, CER 0.022222)
- ref: We walked to the park and watched the children play.
  - forced hyp: We walk to the park and watch the children play. (WER 0.2, CER 0.095238)
- ref: Could you tell me the time and the nearest train station?
  - forced hyp: Could you tell me the time and the nearest train station? (WER 0.0, CER 0.0)

### hi - UNSUPPORTED
_the Kokoro asset has no front-end for this language; it refuses rather than emit romanised gibberish, so no round-trip number exists_

### bn - UNSUPPORTED
_the Kokoro asset has no front-end for this language; it refuses rather than emit romanised gibberish, so no round-trip number exists_

### zh - FAIL
- ref: 今天天气很好。
  - forced hyp: 今天天气很好 (WER 1.0, CER 0.142857)
- ref: 我喜欢学习中文。
  - forced hyp: 我喜歡學習中文 (WER 1.0, CER 0.5)
- ref: 请帮我打开窗户。
  - forced hyp: 先幫我打開窗戶 (WER 1.0, CER 0.625)
- ref: 他在图书馆里看书。
  - forced hyp: 它在圖書館裡看書 (WER 1.0, CER 0.777778)
- ref: 我们的学校非常大而漂亮。
  - forced hyp: 我們的學校非常大而漂亮 (WER 1.0, CER 0.25)
- ref: 妈妈做了可口的饭菜。
  - forced hyp: 媽媽做了可口的粉彩 (WER 1.0, CER 0.5)

## Hardware profiles

Marked measured only for observed peak RSS; still unmeasured thresholds: max_threads, max_audio_buffer_frames, concurrency, allowed_quantizations

mark_measured flips a profile to measured=True but only max_resident_mib was backed by an observation here; the listed thresholds remain provisional seeds and must not be treated as validated.
