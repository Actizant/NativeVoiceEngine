# Multilingual ASR evaluation — 13 user-supplied voice samples (2026-10-13)

Objective: does an alternative ASR model measurably improve recognition across the target languages, using
the actual uploaded samples, without touching the production defaults?

**Short answer: yes, and the gain is concentrated exactly where the platform is weakest.** `faster-whisper-base`
rendered Hindi and Bengali speech in **Perso-Arabic script** while asserting the correct language codes, so a
language-only check reports success on output a Hindi or Bengali reader cannot use. `large-v3` produced
Devanagari and Bengali respectively, and raised mean cue recall from **0.8480 to 0.9562** - at the cost of a
mean real time factor of **2.363 versus 0.231**, and a model load of **10.16 s versus 0.61 s**.

## What was and was not measured

There are **no verified reference transcripts** in this dataset. Filenames carry a truncated quote of the
opening words, which is marketing text, not annotation. Therefore **no WER and no CER is reported anywhere in
this evaluation**, and no accuracy claim is made. What is reported instead is **cue recall**: the longest
common subsequence between the filename quote and the transcript, bounded to [0,1], labelled as a gross-error
and script detector only.

`expected_language` is inferred from the filename's script and speaker name. It is a hint about the sample,
not proof about the model. Every file decoded cleanly, so nothing below is a decoding artefact.

## Inventory — 13 of 13 accessible and decodable

All files were found in `C:\Users\shres\Downloads`, opened read-only, hashed before and after, and **verified
unchanged**. Isolated 16 kHz mono copies were written to `../_multilingual_eval/`, outside the repository.
Decoding and resampling were performed by ffmpeg through PyAV 18.1.0 identically for every file and every
model, so no preprocessing difference can be mistaken for a recognition difference.

| Sample | Container / codec | Rate / ch | Duration | Inferred language | Category | Reference |
|---|---|---|---|---|---|---|
| Energetic-Male-…-18-25 | mp3 / mp3float | 44100 / 1 | 10.08 s | en | English male voice | none |
| Paula-…-18-26 | mp3 / mp3float | 44100 / 1 | 15.15 s | en | English female voice | none |
| 黄丽萍长自述1-…-18-31 | mp3 / mp3float | 44100 / 1 | 16.27 s | zh | Chinese | none |
| 5.28达人-…-18-34 | mp3 / mp3float | 44100 / 1 | 12.98 s | de | German | none |
| Voz-Femenina-Español-…-18-35 | mp3 / mp3float | 44100 / 1 | 13.30 s | es | Spanish | none |
| 元気な女性-…-18-37 | mp3 / mp3float | 44100 / 1 | 18.31 s | ja | Japanese | none |
| Foot-…-18-42 | mp3 / mp3float | 44100 / 1 | 16.20 s | fr | French | none |
| 보이스1-…-18-44 | mp3 / mp3float | 44100 / 1 | 13.56 s | ko | Korean | none |
| بنت-سورية-77-…-18-46 | mp3 / mp3float | 44100 / 1 | 15.10 s | ar | Arabic | none |
| Рената-Литвинова-…-18-48 | mp3 / mp3float | 44100 / 1 | 13.53 s | ru | Russian | none |
| Matilde-…-18-49 | mp3 / mp3float | 44100 / 1 | 15.57 s | pt | Portuguese | none |
| Ultron-2026-10-04-10-20-… | mp3 / mp3float | 44100 / 1 | 14.92 s | hi | Hindi | none |
| voice_preview_ultron (2).wav | wav / pcm_s16le | 44100 / 1 | 34.77 s | **bn, identified here** | see below | none |

**Total: 209.7 s across 13 files.** The brief listed twelve intended categories plus one file to identify. The
thirteenth category asked about conditionally - **Bengali, "if a suitable sample exists" - does exist**, and it
is the unnamed WAV. No sample exists for any other target language, so the remaining ~48 of the ~60 target
languages are entirely unrepresented in this evaluation.

## The unidentified WAV

`voice_preview_ultron (2).wav` names no language, so it was asked to identify itself.

- It is **the same audio as `voice_preview_ultron (2).mp3`** in the same folder: identical declared duration
  (34.769 s), identical frame count when both decode to 16 kHz mono (556,304 frames), mean absolute sample
  delta **0.1 LSB**. The WAV is a re-export of that MP3, not a distinct recording.
- Both baseline passes asserted `bn` (0.876, 0.889) and returned **Arabic-script** text.
- `large-v3` asserted `bn` at **0.405** and returned coherent **Bengali**:
  "আমি প্রায় আকাশ থেকে উলকাপতের কথা ভাবি, মহা প্রলয়, সব কিছুর সমাপ্তি, আর এক নতুন সূচনা"
  — *I often think about meteors falling from the sky, the great dissolution, the end of everything, and a new
  beginning.*
- That is **the same sentence the Hindi file's own filename states in Devanagari** (मैं अक्सर आसमान से गिरती
  हुई उल्काओं के बारे में…). The two files dated 2026-10-04 are one script read in two languages.

**Conclusion, labelled as inference:** the WAV is a Bengali voice preview of the same text as the Hindi sample.
The evidence is writing system, asserted code, matching content and a same-stem sibling - not a verified
transcript. It carries no filename quote, which is why it has no cue score.

## Side-by-side results

`cr` is cue recall, `script` is the writing system the transcript actually arrived in, `rt` is real time factor.

| Sample | lang | base-default (production) | base-int8 | large-v3-int8 |
|---|---|---|---|---|
| Energetic-Male | en | en/latin cr1.00 rt0.26 | en/latin cr1.00 rt0.18 | en/latin cr1.00 rt2.61 |
| Paula | en | en/latin cr1.00 rt0.18 | en/latin cr1.00 rt0.12 | en/latin cr1.00 rt1.87 |
| 黄丽萍长自述1 | zh | zh/cjk **cr0.58** rt0.21 | zh/cjk cr0.83 rt0.14 | zh/cjk **cr0.90** rt2.05 |
| 5.28达人 | de | de/latin cr1.00 rt0.23 | de/latin cr1.00 rt0.15 | de/latin cr1.00 rt2.37 |
| Voz-Femenina | es | es/latin cr0.82 rt0.23 | es/latin cr0.82 rt0.15 | es/latin cr0.82 rt2.19 |
| 元気な女性 | ja | ja/kana cr0.88 rt0.19 | ja/kana cr0.88 rt0.11 | ja/kana **cr0.85** rt1.81 |
| Foot | fr | fr/latin cr1.00 rt0.22 | fr/latin cr1.00 rt0.14 | fr/latin cr1.00 rt2.26 |
| 보이스1 | ko | ko/hangul cr0.97 rt0.26 | ko/hangul cr0.97 rt0.18 | ko/hangul **cr1.00** rt2.54 |
| بنت-سورية | ar | ar/arabic cr0.95 rt0.26 | ar/arabic cr0.95 rt0.17 | ar/arabic **cr1.00** rt2.52 |
| Рената-Литвинова | ru | ru/cyrillic cr0.97 rt0.20 | ru/cyrillic cr0.97 rt0.14 | ru/cyrillic **cr1.00** rt2.13 |
| Matilde | pt | pt/latin cr1.00 rt0.19 | pt/latin cr1.00 rt0.13 | pt/latin cr1.00 rt1.90 |
| Ultron Hindi | hi | hi/**arabic** **cr0.00** rt0.26 | hi/**arabic** cr0.00 rt0.17 | hi/**devanagari** **cr0.90** rt3.23 |
| voice_preview WAV | bn | bn/**arabic** no cue rt0.29 | bn/**arabic** no cue rt0.20 | bn/**bengali** no cue rt3.23 |

| Aggregate | base-default | base-int8 | large-v3-int8 |
|---|---|---|---|
| mean cue recall, 12 cued samples | 0.8480 | 0.8688 | **0.9562** |
| mean real time factor | 0.231 | **0.151** | 2.363 |
| total inference wall time | 49.4 s | 32.4 s | 509.2 s |
| model load | 0.612 s | 0.500 s | 10.157 s |
| peak RSS | 607.9 MiB | 306.0 MiB | **3246.4 MiB** |
| script / language-code mismatches | 2 | 2 | **0** |
| empty transcripts | 0 | 0 | 0 |
| network attempts | 0 | 0 | 0 |

Configuration was identical across the model comparison except where stated: `beam_size=5`, `vad_filter=False`,
`cpu_threads=4`, `local_files_only=True`, language auto-detection on, one pass per model, files processed in the
same order. `base-default` is the production configuration; CTranslate2 reported that the checkpoint's float16
weights were converted to float32 because this backend has no efficient float16 CPU path.

## Findings separated by kind, because they have different owners

**Recognition and rendering.** Hindi is the only categorical failure: the baseline asserted `hi` at 0.816 and
produced Urdu-script text, scoring 0.00 against the Devanagari quote in the same filename. Chinese on the
production float32 pass came back in **traditional** characters against a **simplified** cue and repeated a
phrase — 都會屬於屬於屬於 — recalling 0.58; the int8 pass of the *same weights* produced simplified output at
0.83. Spanish (0.825 in all three) and Japanese (0.88/0.88/0.85) sit below the European pack: the Spanish
transcript rendered "30 años" where the filename quote says "treinta años", which a character subsequence
counts as a mismatch, so that figure is a formatting artefact and not a hearing error.

**Explicitly not recognition errors.** All 13 decoded cleanly. No empty transcripts. No missing sentence
beginnings: `first_segment_start` was 0.0 for every file in every pass **except one**, where `large-v3` began
the Energetic-Male clip at 0.08 s - an 80 ms lead-in trim, not a lost opening. No false language assignments:
12 of 12 asserted codes equalled the inferred expectation under both models, and the one "unknown" was the WAV,
which has no expectation to miss against.

**Segmentation and quantisation instability.** The same audio through the same weights produced different
segment structures: Chinese gave 4 segments at float32, **10 at int8**, and 1 at large-v3, while Korean held 5
in all three. Quantisation alone changed 8 of 13 transcripts outright while leaving all 13 language assertions
identical. Segment count is therefore not a stable observable here, and any per-segment latency claim must name
its quantisation.

**Instrument defects of mine, found and fixed in this pass (A48).** The first cue extractor read the *speaker
prefix* rather than the quoted sentence, so every cue score came out 0.0. The "longest common subsequence" was
a greedy first-match scan that scored a verbatim Chinese transcript at 0.04; the real LCS scores it 0.90. The
printer crashed on cp1252 twice — once in the evaluation tool, once in the comparison tool — both times after
the JSON had been written, so no evidence was lost. All three are corrected and pinned by
`tests/test_multilingual_eval.py` (18 tests, no device, no model).

**Detector gap, recorded not fixed.** The repeated-clause check compares comma-separated clauses, so it did not
see 都會屬於屬於屬於 repeated *inside* one clause. `repeated_clause: false` in these artefacts therefore does not
mean no repetition occurred.

## Provenance of the samples — the finding I was wrong about first

I began this section writing that the filenames carry vendor product text. That is only half right, and the
other half is more important. The filenames are speaker labels; the **transcripts** are the product copy. The
Energetic-Male clip's own recognised text is:

> "Hey there! After months of intensive development, Fish Audio 1.5 has just launched with major improvements
> to expressiveness, stability, and cross-lan…"

So these are **text-to-speech voice previews from a synthesis vendor**, read from scripts about the vendor's own
product — with the exception of the two ULTRON-dated files, which are Hindi and Bengali previews of ULTRON's own
sentence. Every filename timestamp today sits in a two-minute window (18:25 through 18:49), consistent with one
download session.

Two consequences, and they limit every number above:

1. **This is not human speech.** It is machine speech, near-clean, single-take, studio-level. Recognising it
   well says a great deal about model coverage and almost nothing about microphone behaviour — and the
   microphone work in this project already showed human audio at this machine scores far worse (read-aloud WER
   0.12–0.44 against cue recall above 0.82 here).
2. **They are not a licensed corpus.** No licence terms, no consent basis, no redistribution statement, no
   provenance metadata. They were read in place, hashed, and never copied into the repository; only isolated
   16 kHz working copies exist outside it, and only text metrics plus short transcript excerpts are committed
   (A50). They must not become a shipped evaluation set or a training corpus without a decision.

## Hindi and Bengali specifically

The brief asked whether Hindi and Bengali are genuinely supported by the measured results.

- **Hindi: supported by `large-v3`, not by the baseline, on one clip.** The baseline heard the right language
  and wrote the wrong script in both quantisations, scoring 0.00 against its own filename's Devanagari.
  `large-v3` produced Devanagari at 0.90 cue recall. Language probability stayed low for both models (0.82,
  0.84) — this is the least certain of all 13 clips.
- **Bengali: recognised by `large-v3` with correct script and weak confidence.** 0.405 language probability,
  two segments, coherent Bengali whose meaning matches the Hindi clip. The baseline produced Arabic script and
  13 segments from the same 34.8 s.
- Neither claim survives as *validation*. One clip per language, no reference transcript, synthetic source
  audio. The honest statement is: **the baseline is measurably wrong for both languages in a way a
  language-code check cannot see, and `large-v3` is measurably right in a way that needs a real reference set
  before it can be called support.**
- **Synthesis is a separate, unchanged answer**: the approved TTS asset still renders only `en-us`, `en-gb` and
  `zh`, so Hindi and Bengali output remains refused, and this evaluation changes nothing there.

## Latency, memory and the caveat on them

`large-v3` was loaded and run while the machine reported **1.9–2.4 GiB of physical free memory** against a
resident set of **3,246 MiB**. It therefore ran partly from the pagefile, and its real time factors —
1.81 to 3.23 — should be read as *this machine, under this memory pressure*, not as a clean CPU number. The
comparison is still directional and valid: the baseline passes at 306 and 608 MiB never exceeded available
memory, and the gap (2.363 versus 0.151–0.231) is far larger than any plausible paging artefact.

Model load is the number that would matter most in a live product: **10.16 s versus 0.61 s**. At a first-audio
budget measured in seconds, an engine that cannot warm in time is unusable regardless of accuracy, and this
platform's interactive path is already constrained by synthesis latency, not recognition.

## What this does and does not license as a claim

- It **does** support: on these 13 clips `large-v3` produced the correct writing system for Hindi and Bengali
  where the baseline did not, and higher cue recall on every cued sample it beat (zh, ko, ar, ru, hi) at the
  cost of ~15× latency and ~5× memory.
- It **does not** support: any WER or CER figure; "Hindi and Bengali recognition is validated"; multilingual
  coverage beyond the languages present; or **any** claim about human speech, because no clip here is human.
- The **capability registry is unchanged.** A `MEASURED` status requires a measured metric against a verified
  reference, and none exists, so Hindi and Bengali recognition keep their prior declared status and routing
  must keep refusing to act on them. Nothing was registered, promoted, made default, or integrated.
- `large-v3` has **no manifest, no provenance review and no approval** in this project. It is a cached local
  asset used for measurement only; adopting it as a production asset would need its own review, and its RAM
  demand alone would disqualify it from the Lite profile on this class of machine.

## Reproducing

```
python tools/multilingual_eval.py --pass base-default --inventory-only
python tools/multilingual_eval.py --pass base-default
python tools/multilingual_eval.py --pass base-int8
python tools/multilingual_eval.py --pass large-v3-int8
python tools/multilingual_compare.py
python -m pytest tests/test_multilingual_eval.py
```

Artefacts: `artifacts/evaluation/multilingual-inventory.json`, `multilingual-base-default.json`,
`multilingual-base-int8.json`, `multilingual-large-v3-int8.json`, `multilingual-comparison.json`.
