"""Hardware self-test on the real microphone and speakers.

The rest of the suite deliberately runs without devices: bytes shaped like
capture, sinks that record instead of emitting sound. That is the right trade for
a test suite, but it proves the transport and not the hardware, so this tool is
the separate, explicit pass over the real thing.

Nothing happens to the machine unless it is asked for by name. `--record` opens
the microphone, `--play` sends audible sound to the speakers, and both refuse to
run without `--confirm-devices`, because a script that starts recording or makes
noise on its own is not something to discover halfway through.

Writes a JSON artefact so a hardware claim can be dated and re-run.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import threading
import time
from dataclasses import asdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nativevoice.audio.buffer import RingBuffer  # noqa: E402
from nativevoice.audio.io import (  # noqa: E402
    CapturingOutput,
    SounddeviceInput,
    SounddeviceOutput,
    list_devices,
)
from nativevoice.audio.pcm import measure  # noqa: E402
from nativevoice.audio.preprocess import Preprocessor  # noqa: E402
from nativevoice.audio.vad import (  # noqa: E402
    EnergySpeechDetector,
    SegmenterConfig,
    UtteranceSegmenter,
)
from nativevoice.core.audio import AudioBuffer, AudioFormat  # noqa: E402
from nativevoice.core.cancel import CancellationToken  # noqa: E402
from nativevoice.core.errors import DeviceError  # noqa: E402
from nativevoice.core.time import monotonic_ns  # noqa: E402
from nativevoice.hardware import detect  # noqa: E402


def tone(rate: int = 24000, seconds: float = 1.0, frequency: float = 440.0, amplitude: float = 0.18) -> AudioBuffer:
    """A two-second sweep used only to prove the output path moves air."""
    count = int(rate * seconds)
    samples = array.array("h")
    for index in range(count):
        moment = index / rate
        sweep = frequency * (1.0 + 0.35 * moment / seconds)
        edge = min(1.0, moment / 0.02, (seconds - moment) / 0.02)
        samples.append(int(max(-32768, min(32767, amplitude * edge * math.sin(2 * math.pi * sweep * moment) * 32767))))
    return AudioBuffer(samples, AudioFormat(rate, 1))


def record(seconds: float, rate: int, stall_s: float, save_to: Path | None = None) -> dict:
    """Capture from the default input device and describe what came back."""
    source = SounddeviceInput(target_rate=rate, frame_size=int(rate * 0.02), capture_stall_s=stall_s)
    source.open()
    segmenter = UtteranceSegmenter(
        EnergySpeechDetector(format=AudioFormat(rate, 1)),
        SegmenterConfig(minimum_silence_s=0.35, minimum_speech_s=0.2, maximum_speech_s=15.0, padding_s=0.1),
    )
    ring = RingBuffer(AudioFormat(rate, 1), max_frames=rate * 120)
    preprocessor = Preprocessor(target_format=AudioFormat(rate, 1))
    segments = []
    levels = []
    chunk_count = 0
    wall_start = monotonic_ns()
    try:
        deadline = time.monotonic() + seconds
        for chunk in source.frames():
            chunk_count += 1
            processed, report = preprocessor.process(chunk)
            ring.append(processed)
            levels.append(report.output_level)
            for segment in segmenter.push(processed):
                segments.append(
                    {
                        "start_sample": segment.first_sample,
                        "duration_s": round(segment.duration_s, 3),
                        "truncated": segment.truncated,
                        "rms": round(measure(segment.audio).rms, 5) if segment.audio else 0.0,
                    }
                )
            if time.monotonic() >= deadline:
                break
        for segment in segmenter.flush():
            segments.append({"start_sample": segment.first_sample, "duration_s": round(segment.duration_s, 3), "truncated": segment.truncated, "rms": 0.0})
    finally:
        source.close()
    wall_s = (monotonic_ns() - wall_start) / 1e9
    captured = ring.drain_all()
    written: list[str] = []
    if save_to is not None:
        save_to.mkdir(parents=True, exist_ok=True)
        for index, segment in enumerate(segments):
            if segment.audio is None or segment.audio.frame_count == 0:
                continue
            destination = save_to / f"speech_{int(time.time())}_{index:02d}.wav"
            write_wav(str(destination), segment.audio)
            written.append(str(destination))
    return {
        "recordings_written": written,
        "retained": bool(written),
        "requested_seconds": seconds,
        "wall_seconds": round(wall_s, 3),
        "frames_captured": captured.frame_count,
        "audio_seconds": round(captured.duration_s, 3),
        "capture_real_time_factor": round(wall_s / max(0.001, captured.duration_s), 4),
        "chunks": chunk_count,
        "overflow_frames_dropped": ring.overflow_frames,
        "segments": segments,
        "segment_count": len(segments),
        "median_input_level": (sorted(levels)[len(levels) // 2] if levels else 0.0),
        "peak_input_level": round(max(levels), 4) if levels else 0.0,
        "silence_fraction": round(measure(captured).silence_fraction, 4) if captured.frame_count else 1.0,
    }


def play(seconds: float, rate: int, cancel_after_s: float | None) -> dict:
    """Send a sweep to the default output device, optionally interrupting it."""
    output = SounddeviceOutput()
    buffer = tone(rate=rate, seconds=seconds)
    token = CancellationToken()
    emitted_in_full = None
    started = monotonic_ns()
    if cancel_after_s is None:
        emitted = output.play(buffer, token)
    else:
        result = {}

        def work():
            try:
                result["frames"] = output.play(buffer, token)
            except BaseException as error:  # surfaced as a finding, never as a silent zero
                result["error"] = f"{type(error).__name__}: {error}"

        worker = threading.Thread(target=work)
        worker.start()
        time.sleep(cancel_after_s)
        output.stop()
        token.cancel("barge-in")
        worker.join(timeout=5.0)
        emitted = result.get("frames", 0)
        if "error" in result:
            result_entry_error = result["error"]
        emitted_in_full = buffer.frame_count
    elapsed = (monotonic_ns() - started) / 1e9
    recovered = output.play(tone(rate=rate, seconds=0.25), CancellationToken())
    return {
        "requested_frames": buffer.frame_count,
        "emitted_frames": emitted,
        "cancelled_after_s": cancel_after_s,
        "recovery_frames_emitted": recovered,
        "playback_wall_seconds": round(elapsed, 3),
        "device_recovered_after_stop": recovered > 0,
        "full_length_reference": emitted_in_full,
        "thread_error": result.get("error", ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--confirm-devices", action="store_true", help="required: acknowledge that the microphone is opened and sound is played")
    parser.add_argument("--record", type=float, default=0.0, help="seconds to capture from the default input device")
    parser.add_argument("--play", action="store_true", help="play an audible sweep through the default output device")
    parser.add_argument("--cancel-after", type=float, default=0.0, help="interrupt the sweep after this many seconds instead of playing it fully")
    parser.add_argument("--turns", type=int, default=1, help="repeat the record/play cycle this many times")
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--play-seconds", type=float, default=1.0)
    parser.add_argument("--out", default="artifacts/device/selftest.json")
    parser.add_argument("--retention", choices=("none", "keep"), default="none",
                        help="none records and measures without writing any audio to disk")
    parser.add_argument("--recordings-dir", default="",
                        help="where kept recordings go; defaults to the host recordings directory")
    parser.add_argument("--delete-recordings", default="", metavar="DIR",
                        help="remove every WAV under DIR and exit (explicit, auditable purge)")
    args = parser.parse_args(argv)

    if args.delete_recordings:
        target = Path(args.delete_recordings)
        if not target.is_dir():
            print(f"nothing to delete: {target} is not a directory")
            return 1
        removed = [str(path) for path in sorted(target.rglob("*.wav"))]
        for path in target.rglob("*.wav"):
            path.unlink()
        print(json.dumps({"deleted": removed, "directory": str(target)}, indent=2))
        return 0

    if (args.record or args.play) and not args.confirm_devices:
        parser.error("--record and --play touch real hardware; re-run with --confirm-devices")
    if not args.record and not args.play:
        devices = list_devices()
        report = detect()
        print(json.dumps({"devices": len(devices), "inputs": [d.name for d in devices if d.kind == "input"][:3], "gpu": report.gpu_name if hasattr(report, "gpu_name") else str(report)[:120], "note": "no device was opened; pass --record or --play with --confirm-devices"}, indent=2))
        return 0

    result: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": []}
    for turn in range(args.turns):
        entry: dict = {"turn": turn + 1}
        try:
            if args.record:
                destination = None
                if args.retention == "keep":
                    destination = Path(args.recordings_dir) if args.recordings_dir else Path(__import__("ultron_voice.config", fromlist=["data_root"]).data_root()) / "hosts" / "selftest" / "recordings"
                entry["capture"] = record(args.record, args.rate, stall_s=max(1.5, args.record), save_to=destination)
                entry["retention"] = args.retention
            if args.play:
                entry["playback"] = play(args.play_seconds, 24000, args.cancel_after or None)
        except DeviceError as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        result["runs"].append(entry)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"artefact -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
