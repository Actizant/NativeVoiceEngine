# Capability matrix (human summary)

The authoritative, machine-generated version is `artifacts/evaluation/capability-matrix.md`; this page is
how to read it. Generated 2026-10-12.

| Language | Recognition | Synthesis | Evidence |
|---|---|---|---|
| English | **MEASURED** — round-trip WER 0.0494, CER 0.0242 | **MEASURED** — RTF 1.55–2.38, first audio 3.9–8.3 s | `artifacts/evaluation/roundtrip-2026-10-12.json` |
| Chinese | declared only (vendor model card) | **MEASURED-FAILED** — CER 0.463 against a 0.10 bar | same artefact, the `!` cell |
| Hindi | declared only | **UNSUPPORTED** — no front-end in the approved asset | refusal recorded verbatim |
| Bengali | declared only | **UNSUPPORTED** — no front-end in the approved asset | refusal recorded verbatim |
| the other 57 target languages | `NO_MODEL` | `NO_MODEL` | registry default: unrecorded never means supported |

## Reading the symbols

`M` measured here · `d` declared by upstream and unverified · `P` partial · `!` measured and failed the bar ·
`x` unsupported (refused, never attempted) · `-` awaiting an approved model.

`!` and `x` are deliberately different statuses. One says the asset exists and the audio is bad; the other
says no asset exists. They imply completely different work, and merging them is how a capability report
stops being actionable. Routing treats both as unusable; reporting keeps them apart.

## Limits that apply to every number above

1. **Round-trip metrics measure the TTS+STT pair on machine-generated audio.** No licensed human-speech
   corpus exists on this machine, and `eval/samples.py` refuses synthetic fixtures in a quality path, so
   English WER 0.0494 is a floor on intelligibility rather than a measurement of it.
2. **Chinese figures are inflated** by a simplified-vs-traditional mismatch with no approved converter to
   normalise either side. Character error rate is primary because word error rate is meaningless on an
   unsegmented script — a whole sentence counts as one token, so one wrong character reported WER 1.000.
3. **Latency comes from two runs that disagree** (4–8 s first audio). Both are retained and labelled;
   neither was quietly dropped in favour of the flattering number.
4. **No hardware profile is validated.** Peak RSS is measured (1,001 / 1,015 / 1,046 MiB at 2 / 4 / 8
   threads), which is one of five threshold fields. Buffer size, concurrency and quantization policy are
   unbenchmarked, so all three profiles still report `provisional: True`.
5. **Recognition for Hindi and Bengali is unmeasured**, so no routing decision may claim it and no report
   here should be read as saying those languages work.
