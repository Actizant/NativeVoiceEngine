# Real-speech validation protocol — microphone, human speech, retention and deletion

**Nothing in this document has been executed.** Recording from a microphone is a physical action in
Shreshtha's space, so it waits for his explicit confirmation, one run at a time. Everything already
verified on synthetic bytes and on silent/room-noise capture stays as it is; this protocol exists to
close the two claims that synthetic audio cannot support — VAD on a real voice, and recognition accuracy
on human speech — and to stop anyone later reading a round-trip number as a human benchmark.

## What is already proven without a human

- capture sustains real time (3.98 s of audio in 4.099 s wall, 199 chunks, zero frames dropped);
- the VAD opens **no** turn on 4 s of a real, quiet room (median level 0.0003, peak 0.0019);
- playback, mid-span barge-in (12,000 of 72,000 frames then halt) and device recovery all work on the real
  output device;
- segmentation, endpointing, cancellation and turn accounting are measured on synthetic envelopes.

## What a human is required for

| Claim | Why synthetic cannot settle it | Metric recorded |
|---|---|---|
| VAD detects real speech, at real levels, in a real room | Room acoustics, distance, mic gain and breath/pause patterns are not in the fixture | onset error (ms), offsets, false-open count over 60 s of ambient with speech inserted |
| Recognition accuracy on human speech | A round-trip measures our TTS against our ASR, not speech | WER and CER per language, with detected-language accuracy |
| Language identification on speech | Text-domain LID never sees audio | top-1 and refusal rate at the configured threshold |
| Barge-in against a real voice | A tone never starts mid-word | interrupt latency, whether stale audio escaped |

## Recording location, retention and deletion — actual implemented behaviour

`tools/selftest_audio.py` (implemented, gated):

- **Default retention is `none`.** The run measures levels, segmentation and timing, and **writes no audio
  to disk at all**. The capture exists only in memory during the process.
- `--retention keep` writes **detected speech segments** (not the raw room audio) to
  `--recordings-dir DIR`, defaulting to `<data root>\hosts\selftest\recordings`, i.e.
  `%LOCALAPPDATA%\ultron-voice\hosts\selftest\recordings` — the same local root as everything else, never
  a temp directory that outlives the run unnoticed, and never outside this machine.
- Naming: `speech_<unix-time>_<index>.wav`, 16-bit PCM, so a file's timestamp is in its name.
- Deletion is explicit and verifiable:
  `python tools/selftest_audio.py --delete-recordings "<dir>"` lists every path removed in the printed
  JSON and refuses if the directory does not exist. There is no background cleanup and no silent expiry.
- Nothing is ever uploaded. `tools/audit.py` proves no network module is importable on the speech path, and
  `tools/offline_proof.py` proves a full synthesis→recognition round-trip makes zero connection attempts.

The consent model is the one already in the engine: a recording that becomes a **voice profile** requires a
`ConsentRecord` before it can be used, and a revoked consent makes the profile unusable. A transient
recognition test with `--retention none` never becomes a profile and never touches that store.

## The procedure, once he confirms

Run each step separately so the machine is otherwise quiet and he can hear when it is listening.

```bash
# 1. baseline room, no speech expected — proves the gate does not open on its own
python tools/selftest_audio.py --record 8 --rate 16000 --confirm-devices

# 2. read-aloud, English, three sentences at ~30 cm, natural pace
python tools/selftest_audio.py --record 25 --rate 16000 --retention keep --confirm-devices

# 3. pause-heavy speech (the failure mode the reference stack documented)
#    speak a sentence, stay silent 1.5 s, speak again — endpointing must produce 2 turns

# 4. background noise (fan / music at conversational level), then speech over it

# 5. Hindi, then Bengali, read-aloud — recognition and language detection per utterance

# 6. interruption: start a spoken answer, speak over it, confirm playback halts
python tools/selftest_audio.py --play --play-seconds 4 --cancel-after 1.2 --confirm-devices

# 7. erase everything kept
python tools/selftest_audio.py --delete-recordings "%LOCALAPPDATA%\\ultron-voice\\hosts\\selftest\\recordings"
```

Read-aloud is stated as read-aloud wherever its numbers appear. Read speech is cleaner than spontaneous
speech and biases WER optimistically; that caveat travels with the result, and no read-aloud figure is
ever written into the capability matrix as a general human-speech benchmark.

## Acceptance criteria proposed, agreed before recording

| Criterion | Bar | Basis |
|---|---|---|
| False turn-opens on quiet room | 0 over 30 s | already met on 4 s |
| VAD onset error on speech | ≤ 150 ms | fixture test currently holds 60 ms |
| Endpointing on 1.5 s pauses | 2 turns, not 1 | policy test exists; real-audio case open |
| English read-speech WER (whisper-base) | report, no bar set | so the number is not chosen after seeing it |
| Language detection, English vs Hindi vs Bengali | report top-1 and refusals; refusals are not errors | a refusal at low confidence is the specified behaviour |
| Barge-in | no audio after cancel, no turn leak | measured on device for tones |

Reporting after each run: the JSON artefact under `artifacts/device/`, free RAM and process count at run
time, the exact device index and rate used, and any clip kept — so a claim can be re-run or deleted.

## Results — step 2, read-aloud English (attempted 2026-10-13, no speech captured)

Command actually run, from the repository root, retention defaulting to `none`:

```
python tools/real_speech_check.py --countdown 12
```

Artefact: `artifacts/device/real-speech-read-aloud.json`. Targets were the three predeclared sentences
(the pangram, the lights sentence, the coffee-and-bread sentence), printed before capture so the reading
is verifiable rather than reconstructed afterwards.

| Criterion (predeclared above) | Result | Evidence |
|---|---|---|
| No audio persisted | **PASS** | 381 files before, 381 after, `created_during_run: []`, `recordings_written: []` |
| Processed locally, offline | **PASS** | `network_attempts: []` under the runtime `OfflineGuard` |
| False turn-opens on quiet room | **PASS** | 0 openings in the quiet window, median input level 0.000349, peak 0.0044 |
| All sentences detected | **FAIL** | 0 segments over 74.98 s of captured audio; all three targets `MISSING_SEGMENT` |
| Read-aloud WER / CER | **NOT RUN** | nothing to score; `sentences_scored: 0`, both means `null` |
| Language detection top-1 | **NOT RUN** | `language_detection_top1: null` |
| VAD onset error ≤ 150 ms | **NOT RUN** | and structurally unreachable this way — a manual read-aloud run has no ground-truth onset timestamp to measure against; it needs a keyed stimulus, not a countdown |
| Endpointing on 1.5 s pauses | **NOT RUN** | step 3 |
| Noise, barge-in | **NOT RUN** | steps 4 and 6 |

The failure is an absence of input, not a defect in the capture chain: the same harness on the same
device recorded 4 s of clean quiet-room audio twice in the previous session, the VAD opened and closed
correctly on synthetic tones, and the offline and persistence gates held. Peak level across the whole
75 s window was 0.0044 — the same order as the documented quiet-room baseline, so nothing that could
be called speech reached the microphone. The honest reading is that **no speaker was present during the
window** (the run was launched unattended to keep this turn inside the authorised step).

Therefore the two claims the master prompt says only real speech can prove — **recognition quality on
human audio** and **language detection on human audio** — remain unproven, and `status: NOT RUN` is
where they stay in the artefact. Step 2 is not closed. It needs one rerun with someone actually reading
the three sentences into the confirmed default input; the harness is ready and needs no changes, only a
voice.

Numbers from this run are kept out of the synthetic benchmark set entirely, and the synthetic
round-trip figures (English WER 0.0494 / CER 0.0242) are not evidence about human speech.

## Results — step 2, first run with a speaker present (2026-10-13 14:51:57)

Same command, longer countdown because the reader was in the room:

```
python tools/real_speech_check.py --countdown 20
```

Preserved byte-for-byte at `artifacts/device/real-speech-read-aloud.20261013-145157.human-run.original.json`.
**The section above this one stands as what was observed when it was written; this run supersedes its
conclusion** that no speech could be captured.

Real speech reached the chain: peak input level **0.0866** against the documented quiet-room 0.0044, about
twenty times room tone. The detector opened a turn at audio-time **21.34 s** holding **3.8 s** of speech,
and Whisper transcribed it **word for word**:

| Against the sentence actually spoken | WER | CER |
|---|---|---|
| "I would like a cup of coffee and a slice of bread." | **0.0000** | **0.0000** |
| the pangram (what the harness scored it against) | 1.3333 | 0.9714 |
| "Please turn off the lights..." | 1.2000 | 0.8837 |

Language detection returned `en` at confidence **1.000** on that human utterance. Nothing was persisted
(381 files before and after, `created_during_run` empty) and the offline guard recorded no attempts.

Three of the four claims this protocol exists to test are therefore now demonstrated, once each, on human
audio: VAD opening on a voice, recognition of a spoken sentence, and language identification of it.

What this run did **not** settle is endpointing on real pauses, because the harness itself was broken in
four ways, all found by reading its output against its source:

| Finding | What the harness did |
|---|---|
| A35 | It paired `segments[index]` with `SENTENCES[index]`, so one utterance was scored against the first target whichever sentence it actually was — inventing WER 1.3333 for a perfect transcript |
| A36 | Speech detected inside the two-second post-countdown "quiet baseline" went to a list that was measured and then discarded; two openings were swallowed and simultaneously counted as *false* turn-opens |
| A37 | `quiet_window_seconds` reported one 20 ms chunk's duration instead of the window that was measured, so the artefact claimed a 0.02 s baseline |
| A38 | `segment_onsets[].seconds` carried the segment *duration*, not its onset — which is why an earlier report said the utterance began at 3.8 s when it began at 21.34 s |

So `segments_detected: 1` of 3 is **not** evidence about endpointing on pauses: with two openings discarded
by A36, the run is consistent with all three sentences having been spoken. After the harness corrections the
criterion is re-tested from a clean base, and until that rerun happens endpointing on human pauses stays
`NOT RUN`, not failed and certainly not passed.

Corrected harness behaviour, in words: sentences are matched to recognised content by minimum word error
with a refusal cap, so missing, extra and duplicate utterances are named rather than forced into a pairing;
the quiet window measures levels on the same stream that produces segments, so nothing spoken is dropped;
and every turn opening is logged with its onset in audio seconds, marked for whether it fell inside the
quiet window and whether it claimed a sentence — a *false* opening is one inside the window that claimed
nothing.


## Results — step 2, supervised rerun with the corrected harness (2026-10-13 15:31:33)

```
python tools/real_speech_check.py --countdown 60 --limit 100
```

Artefact `artifacts/device/real-speech-read-aloud.json`, 5,900 bytes. The 15:18 run is preserved at
`real-speech-read-aloud.20261013-151857.previous.json` and the 14:51 run at
`real-speech-read-aloud.20261013-145157.human-run.original.json` - three generations of a measurement that
has been corrected twice, all kept.

| Criterion | Result |
|---|---|
| Nothing persisted | **PASS** - retention `none`, 381 files before and after, no recordings written |
| Offline | **PASS** - `network_attempts` empty |
| False turn-opens | **PASS** - 0 openings in the quiet window, measured 2.0 s, room median 0.000632, peak 0.0019 |
| Sentences matched | **FAIL** - 2 of 3 |
| Read-aloud WER / CER | **MEASURED** - mean **0.3084 / 0.2408**, 0 exact matches |
| Language detection | **MEASURED** - 1.000 on this run (`en` at 0.9782 and 0.9557) |
| Onset error, endpointing, noise, barge-in | **NOT RUN** |

What came back, word for word:

- "Please turn off the lights **before you have the**" - WER 0.2000, 8 correct, 1 substitution, 1 deletion
- "**Cut off** of coffee and a slice of bread." - WER 0.4167, 7 correct, 2 substitutions, 3 deletions
- The pangram arrived as **two fragments 0.2 s apart** - "The quick brown walks." and "Johns water leasing." -
  scoring 0.6667 and 1.000 against it, so the 0.60 alignment cap refused both and reported them as extras.
  The content matcher did its job: an honest `FAIL` replaced the fake number the positional code would
  have produced.
- One fragment between sentences hallucinated "アイルーブ ライカー" - the classic Whisper response to audio it
  cannot parse.

### The finding this run produced about itself: A42

`wall_seconds` came back **100.136** for a run of 60 s countdown plus up to 100 s of capture. That only
adds up if the countdown's audio was replayed instantly - and it was. `SounddeviceInput.open()` starts a
callback that queues everything it receives, and the previous code opened the stream **before** counting
down, so about a minute of pre-GO silence entered the analysis at CPU speed. Consequences, all real:

1. the speaker's usable window after the GO cue was roughly **40 s, not the 100 s promised**;
2. the quiet baseline measured the room **during the countdown**, so "0 false openings" was judged against
   the wrong interval;
3. onset timestamps carried the backlog - the first opening at 77.06 s was actually ~17 s after GO.

Fixed by opening the stream only after the cue, and now tested against a device double that refuses to
yield frames while it is closed. `wall_seconds` measures the capture alone; the artefact records
`countdown_seconds` and `stream_opened_at`.

A40 was reproduced exactly as predicted: the run crashed on the **final console print** of the Japanese
fragment, after the artefact was safely written. Also fixed - UTF-8 reconfiguration plus an `emit()` that
degrades to ASCII escapes and can never abort a report, with `write_artefact()` ordered before any printing.

### Standing position on step 2

VAD opening on a human voice: **proven**. Recognition of human speech: **proven but variable** - one
sentence word-exact (14:51), two sentences at 0.20 and 0.42 WER (15:31), and the synthetic round-trip
figure of 0.0494 remains a synthetic result only. Language detection through the vendor's auto-detect:
**unstable** (0.9467 confidence on `nn` for English in one run, correct `en` twice in the next) - logged as
A41, open, not investigated. Endpointing on real pauses: **still NOT RUN**, because the one run that would
have answered it was confounded by A36 and the run after that by A42. A rerun with both fixed is the next
measurement, and it is the reader's to trigger.
