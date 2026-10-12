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
