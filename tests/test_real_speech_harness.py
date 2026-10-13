"""Regression tests for the read-aloud harness and the A/B player.

These cover findings A33-A37: sentences matched by content instead of by position, the quiet-room
measurement no longer swallowing speech, the quiet window reporting the time it actually measured,
turn openings logged with their onset, and playback order declared rather than derived from filenames.

Nothing here touches a microphone, a loudspeaker or a model - the capture policy takes an iterable of
chunks and a duck-typed segmenter, so it is testable on synthetic audio.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fixtures import silence, speech_like

from nativevoice.audio.buffer import RingBuffer
from nativevoice.audio.preprocess import Preprocessor
from nativevoice.audio.vad import EnergySpeechDetector, SegmenterConfig, UtteranceSegmenter
from nativevoice.core.audio import AudioFormat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.real_speech_check import (  # noqa: E402
    ALIGN_WER_CAP,
    SENTENCES,
    align_targets,
    classify_openings,
    collect_segments,
    configure_console,
    emit,
    emit_block,
    open_and_capture,
    write_artefact,
)

RATE = 16000


def segment(first_sample: int, duration_s: float, truncated: bool = False) -> SimpleNamespace:
    """The three fields `classify_openings` reads off a real SpeechSegment."""
    return SimpleNamespace(first_sample=first_sample, duration_s=duration_s, truncated=truncated)


def collect(signal, *, quiet_window_s: float, limit_s: float = 30.0, stop_at: int = 6) -> dict:
    fmt = AudioFormat(RATE, 1)
    return collect_segments(
        chunks=signal.frames(320),
        pre=Preprocessor(target_format=fmt),
        segmenter=UtteranceSegmenter(
            EnergySpeechDetector(format=fmt),
            SegmenterConfig(minimum_silence_s=0.05, minimum_speech_s=0.05, padding_s=0.02),
        ),
        ring=RingBuffer(fmt, max_frames=RATE * 60),
        quiet_window_s=quiet_window_s,
        limit_s=limit_s,
        stop_at_segments=stop_at,
    )


class TestContentAlignment:
    """A35: a segment was scored against whichever target shared its index."""

    def test_the_cap_is_the_documented_distance(self):
        assert ALIGN_WER_CAP == 0.60

    def test_a_sentence_spoken_out_of_order_is_matched_to_itself(self):
        # exactly the run that exposed the defect: one utterance, the third sentence, word for word
        result = align_targets(list(SENTENCES), [SENTENCES[2]])
        assert [(item["target_index"], item["segment_index"]) for item in result["matches"]] == [(2, 0)]
        assert result["matches"][0]["wer"] == 0.0

    def test_the_unspoken_targets_are_missing_rather_than_padded(self):
        result = align_targets(list(SENTENCES), [SENTENCES[2]])
        assert {item["target_index"] for item in result["missing_targets"]} == {0, 1}
        assert all(item["closest_wer"] > ALIGN_WER_CAP for item in result["missing_targets"])
        assert result["extra_segments"] == []

    def test_three_sentences_in_a_scrambled_order_still_pair_correctly(self):
        result = align_targets(list(SENTENCES), [SENTENCES[2], SENTENCES[0], SENTENCES[1]])
        pairs = {item["target_index"]: item["segment_index"] for item in result["matches"]}
        assert pairs == {2: 0, 0: 1, 1: 2}
        assert all(item["wer"] == 0.0 for item in result["matches"])

    def test_a_partial_sentence_still_claims_its_target(self):
        heard = "The quick brown fox jumps over the lazy."
        result = align_targets(list(SENTENCES), [heard])
        assert result["matches"][0]["target_index"] == 0
        assert 0.0 < result["matches"][0]["wer"] < ALIGN_WER_CAP

    def test_unrelated_speech_is_reported_as_extra_not_forced_onto_a_target(self):
        heard = [SENTENCES[0], "the cat slept on a roof garden last Monday"]
        result = align_targets(list(SENTENCES), heard)
        assert [item["target_index"] for item in result["matches"]] == [0]
        assert len(result["extra_segments"]) == 1
        extra = result["extra_segments"][0]
        assert extra["segment_index"] == 1
        assert extra["reason"] == "recognised speech that matches no target sentence"

    def test_a_repeated_sentence_leaves_the_second_copy_unclaimed(self):
        result = align_targets(list(SENTENCES), [SENTENCES[0], SENTENCES[0]])
        assert [(item["target_index"], item["segment_index"]) for item in result["matches"]] == [(0, 0)]
        assert result["extra_segments"][0]["reason"] == "duplicate of sentence 1"

    def test_nothing_is_claimed_twice(self):
        result = align_targets(list(SENTENCES), list(SENTENCES) + list(SENTENCES))
        assert len(result["matches"]) == len(SENTENCES)
        assert len({item["target_index"] for item in result["matches"]}) == len(SENTENCES)
        assert len({item["segment_index"] for item in result["matches"]}) == len(SENTENCES)

    def test_alignment_is_deterministic(self):
        heard = [SENTENCES[1], SENTENCES[2], SENTENCES[0]]
        first = align_targets(list(SENTENCES), heard)
        second = align_targets(list(SENTENCES), heard)
        assert first["matches"] == second["matches"]
        assert first["extra_segments"] == second["extra_segments"]

    def test_no_speech_at_all_matches_everything_as_missing(self):
        result = align_targets(list(SENTENCES), [])
        assert result["matches"] == []
        assert len(result["missing_targets"]) == len(SENTENCES)
        assert all(item["closest_wer"] is None for item in result["missing_targets"])


class TestQuietWindowKeepsSpeech:
    """A36: openings inside the measurement window were filed as noise and thrown away."""

    def test_speech_beginning_during_the_quiet_window_is_captured(self):
        stats = collect(speech_like(rate=RATE, bursts=((0.05, 0.25), (0.6, 0.8))), quiet_window_s=0.4)
        assert len(stats["segments"]) == 2
        onsets = [item.first_sample / RATE for item in stats["segments"]]
        assert onsets[0] < 0.4

    def test_speech_beginning_immediately_after_the_countdown_is_captured(self):
        stats = collect(speech_like(rate=RATE, bursts=((0.0, 0.2),)), quiet_window_s=0.05)
        assert len(stats["segments"]) == 1
        assert stats["segments"][0].first_sample / RATE < 0.05

    def test_quiet_room_alone_produces_no_segments_but_still_measures(self):
        stats = collect(silence(rate=RATE, seconds=1.0), quiet_window_s=0.4)
        assert stats["segments"] == []
        assert len(stats["quiet_levels"]) > 0
        assert pytest.approx(stats["quiet_seconds"], abs=0.03) == 0.4

    def test_the_capture_stops_at_the_declared_opening_count(self):
        stats = collect(
            speech_like(rate=RATE, bursts=((0.2, 0.4), (0.6, 0.8), (1.0, 1.2))),
            quiet_window_s=0.1,
            stop_at=2,
        )
        assert len(stats["segments"]) == 2
        assert stats["stopped_because"] == "reached 2 turn openings"

    def test_the_capture_limit_is_honoured_in_audio_seconds(self):
        stats = collect(silence(rate=RATE, seconds=3.0), quiet_window_s=0.2, limit_s=1.0, stop_at=99)
        assert stats["audio_seconds"] <= 1.03
        assert "capture limit" in stats["stopped_because"]


class TestOpeningLog:
    """A36 and A37: every opening logged with its onset, and false only when nothing was claimed."""

    def test_onset_is_audio_time_and_duration_stays_separate(self):
        report = classify_openings([segment(341440, 3.8)], RATE, 2.0, {0})
        opening = report["openings"][0]
        assert opening["onset_seconds"] == pytest.approx(21.34, abs=0.01)
        assert opening["duration_seconds"] == pytest.approx(3.8)
        assert opening["within_quiet_window"] is False

    def test_speech_inside_the_window_is_genuine_speech_not_a_false_opening(self):
        report = classify_openings([segment(1600, 1.2)], RATE, 2.0, {0})
        assert report["openings"][0]["within_quiet_window"] is True
        assert report["false_turn_openings"] == 0
        assert report["speech_inside_quiet_window"] == 1

    def test_only_an_unclaimed_opening_inside_the_window_is_false(self):
        report = classify_openings([segment(3200, 0.3), segment(48000, 2.0)], RATE, 2.0, {1})
        assert report["openings_in_quiet_window"] == 1
        assert report["false_turn_openings"] == 1
        assert report["speech_inside_quiet_window"] == 0

    def test_an_opening_after_the_window_is_never_counted_false(self):
        report = classify_openings([segment(80000, 0.3)], RATE, 2.0, set())
        assert report["false_turn_openings"] == 0

    def test_truncation_is_carried_through_to_the_log(self):
        report = classify_openings([segment(1600, 12.0, truncated=True)], RATE, 2.0, {0})
        assert report["openings"][0]["truncated"] is True


class TestCountdownCue:
    """The countdown is the start signal for a person in the room, so it has to survive the console."""

    def test_every_tick_is_its_own_line_and_the_go_marker_is_loud(self, capsys, monkeypatch):
        import tools.real_speech_check as harness

        monkeypatch.setattr(harness.time, "sleep", lambda _seconds: None)
        harness.countdown(3)
        lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
        assert any(line.startswith("window opens in 3 seconds") for line in lines)
        assert [line for line in lines if line.startswith("...")] == ["... 3", "... 2", "... 1"]
        assert any(line.startswith("GO - START READING NOW") for line in lines)

    def test_the_cue_never_depends_on_overwriting_one_line(self, capsys, monkeypatch):
        import tools.real_speech_check as harness

        monkeypatch.setattr(harness.time, "sleep", lambda _seconds: None)
        harness.countdown(2)
        raw = capsys.readouterr().out
        assert "\r" not in raw, "a carriage-return tick arrives as one mashed line, after the window opened"
        assert raw.count("\n") >= 6

    def test_go_marker_is_the_last_thing_printed_before_capture(self, capsys, monkeypatch):
        import tools.real_speech_check as harness

        monkeypatch.setattr(harness.time, "sleep", lambda _seconds: None)
        harness.countdown(1)
        lines = [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert lines[-1] == "=" * 58
        assert lines[-2].startswith("GO - START READING NOW")


class TestAbPlaybackOrder:
    """A33 and A34: the switch that was never implemented, and an order that came from sorting."""

    def test_incumbent_is_played_before_the_candidate(self, tmp_path):
        from tools.ab_samples import pair_clips

        for name in ("01_kitten_punctuation.wav", "01_kokoro_punctuation.wav"):
            (tmp_path / name).write_bytes(b"RIFF")
        assert [engine for engine, _ in pair_clips(tmp_path, 1)] == ["kokoro", "kitten"]

    def test_labels_name_the_engine_and_its_role(self):
        from tools.ab_samples import LABELS, PLAYBACK_ORDER

        assert PLAYBACK_ORDER == ("kokoro", "kitten")
        assert LABELS["kokoro"].startswith("A") and "Kokoro" in LABELS["kokoro"]
        assert LABELS["kitten"].startswith("B") and "Kitten" in LABELS["kitten"]

    def test_a_pair_missing_one_clip_still_plays_what_exists(self, tmp_path):
        from tools.ab_samples import pair_clips

        (tmp_path / "03_kokoro_long-utterance.wav").write_bytes(b"RIFF")
        assert [engine for engine, _ in pair_clips(tmp_path, 3)] == ["kokoro"]
        assert pair_clips(tmp_path, 4) == []

    def test_pair_numbers_match_the_rendered_files(self, tmp_path):
        from tools.ab_samples import SENTENCES, pair_clips

        for index in range(1, len(SENTENCES) + 1):
            for engine in ("kitten", "kokoro"):
                (tmp_path / f"{index:02d}_{engine}_slug.wav").write_bytes(b"RIFF")
        for index in range(1, len(SENTENCES) + 1):
            assert [engine for engine, _ in pair_clips(tmp_path, index)] == ["kokoro", "kitten"]

    def test_the_player_refuses_a_non_numeric_pair(self):
        from tools.ab_samples import play_argument

        with pytest.raises(ValueError, match="pair number"):
            play_argument(["tools/ab_samples.py", "--play", "all"])

    def test_the_player_refuses_a_missing_pair_number(self):
        from tools.ab_samples import play_argument

        with pytest.raises(ValueError, match="pair number"):
            play_argument(["tools/ab_samples.py", "--play"])

    def test_no_all_pairs_switch_is_advertised_anywhere_in_the_tool(self):
        source = (ROOT / "tools" / "ab_samples.py").read_text(encoding="utf-8")
        assert "--play all" not in source


def test_a_previous_artefact_is_copied_before_a_run_replaces_it(tmp_path):
    """The corrected rerun must not erase the evidence it corrects."""
    from tools.real_speech_check import preserve_previous

    artefact = tmp_path / "real-speech-read-aloud.json"
    artefact.write_text('{"generated": "2026-10-13T14:51:59"}', encoding="utf-8")
    kept = preserve_previous(artefact)
    assert kept and Path(kept).exists()
    assert Path(kept).name == "real-speech-read-aloud.20261013-145159.previous.json"
    assert "14:51" in Path(kept).read_text(encoding="utf-8")

    # a second rerun keeps its own generation rather than overwriting the first backup
    artefact.write_text('{"generated": "2026-10-13T16:05:00"}', encoding="utf-8")
    second = preserve_previous(artefact)
    assert second != kept and Path(kept).exists() and Path(second).exists()

    assert preserve_previous(tmp_path / "nothing-here.json") == ""


class FakeStream:
    """A console that cannot encode what the recogniser heard, mirroring a cp1252 Windows terminal."""

    def __init__(self, encoding: str = "ascii"):
        self.buffer = io.BytesIO()
        self.stream = io.TextIOWrapper(self.buffer, encoding=encoding, errors="strict")

    def replace(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", self.stream)
        return self

    def written(self) -> str:
        self.stream.flush()
        return self.buffer.getvalue().decode("ascii", "replace")


class TestConsoleGuard:
    """A40: a Japanese or Cyrillic hallucination crashed the report *after* the evidence was saved."""

    def test_emit_reprints_as_ascii_when_the_console_cannot_encode_it(self, monkeypatch):
        console = FakeStream().replace(monkeypatch)
        emit("  extra : \u30a2\u30a4\u30eb\u30fc\u30d6 \u30e9\u30a4\u30ab\u30fc")
        text = console.written()
        assert "extra :" in text
        assert text.isascii(), "the fallback must reprint in ASCII rather than lose the line"
        assert chr(92) + "u30a2" in text, "escape the undisplayable characters, never drop them"

    def test_emit_survives_a_stream_that_raises_on_write(self, monkeypatch):
        class Dead:
            def write(self, *_args):
                raise OSError("stream closed")

            def flush(self):
                raise OSError("stream closed")

        monkeypatch.setattr(sys, "stdout", Dead())
        emit("anything at all")  # must not raise: the artefact, not the console, is the record

    def test_configure_console_tolerates_a_stream_it_cannot_reconfigure(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", object())
        configure_console()  # a captured or foreign stream simply keeps the emit() fallback

    def test_emit_block_prints_every_line_even_when_one_cannot_be_encoded(self, monkeypatch):
        console = FakeStream().replace(monkeypatch)
        emit_block("line one\n\u30a2 line two\nline three")
        text = console.written()
        assert "line one" in text and "line three" in text
        assert text.isascii() and chr(92) + "u30a2" in text


class TestArtefactOutlivesTheConsole:
    """The JSON is the evidence; printing it is a courtesy that must never be able to undo it."""

    def test_the_artefact_is_complete_before_anything_is_printed(self, tmp_path, monkeypatch):
        path = tmp_path / "run.json"
        report = {"generated": "2026-10-13T15:31:33", "extra": "\u30a2\u30a4\u30eb\u30fc\u30d6 \u30e9\u30a4\u30ab\u30fc"}
        assert write_artefact(report, path) == ""
        assert json.loads(path.read_text(encoding="utf-8")) == report

        console = FakeStream().replace(monkeypatch)
        emit_block(json.dumps(report, ensure_ascii=False, indent=2))
        assert json.loads(path.read_text(encoding="utf-8")) == report
        assert console.written().isascii() and chr(92) + "u30a2" in console.written()

    def test_a_replaced_artefact_is_kept_under_its_own_timestamp(self, tmp_path):
        path = tmp_path / "run.json"
        write_artefact({"generated": "2026-10-13T15:18:57"}, path)
        kept = write_artefact({"generated": "2026-10-13T15:31:33"}, path)
        assert Path(kept).name == "run.20261013-151857.previous.json"
        assert json.loads(Path(kept).read_text(encoding="utf-8"))["generated"] == "2026-10-13T15:18:57"


class OrderedSource:
    """A device that produces audio only while it is open, the way PortAudio's callback really behaves."""

    def __init__(self, events: list[str], chunks):
        self._events = events
        self._chunks = list(chunks)
        self._open = False

    def open(self) -> None:
        self._events.append("open")
        self._open = True

    def frames(self):
        self._events.append("frames")
        if not self._open:
            raise AssertionError("frames were asked for while the stream was closed")
        return iter(self._chunks)

    def close(self) -> None:
        self._events.append("close")
        self._open = False


def pipeline():
    fmt = AudioFormat(RATE, 1)
    return (
        Preprocessor(target_format=fmt),
        UtteranceSegmenter(
            EnergySpeechDetector(format=fmt),
            SegmenterConfig(minimum_silence_s=0.05, minimum_speech_s=0.05, padding_s=0.02),
        ),
        RingBuffer(fmt, max_frames=RATE * 60),
    )


class TestStartTiming:
    """A42: opening the stream before the countdown queued minutes of pre-GO silence into the analysis."""

    def test_the_stream_opens_only_after_the_go_cue(self):
        events: list[str] = []
        pre, segmenter, ring = pipeline()
        source = OrderedSource(events, silence(rate=RATE, seconds=0.6).frames(320))
        stats = open_and_capture(
            source=source,
            pre=pre,
            segmenter=segmenter,
            ring=ring,
            quiet_window_s=0.2,
            limit_s=5.0,
            stop_at_segments=99,
            on_ready=lambda: events.append("GO"),
        )
        assert events == ["GO", "open", "frames", "close"]
        assert stats["audio_seconds"] <= 0.62, "only post-GO audio may enter the analysis"

    def test_the_quiet_baseline_begins_at_go_not_at_device_open(self):
        events: list[str] = []
        pre, segmenter, ring = pipeline()
        source = OrderedSource(events, speech_like(rate=RATE, bursts=((0.9, 1.1),)).frames(320))
        stats = open_and_capture(
            source=source,
            pre=pre,
            segmenter=segmenter,
            ring=ring,
            quiet_window_s=0.4,
            limit_s=5.0,
            stop_at_segments=99,
            on_ready=lambda: events.append("GO"),
        )
        assert pytest.approx(stats["quiet_seconds"], abs=0.03) == 0.4
        report = classify_openings(stats["segments"], RATE, stats["quiet_seconds"], set())
        assert report["openings"][0]["onset_seconds"] > 0.4
        assert report["openings"][0]["within_quiet_window"] is False

    def test_wall_seconds_measures_the_capture_and_not_the_countdown(self):
        events: list[str] = []
        pre, segmenter, ring = pipeline()
        source = OrderedSource(events, silence(rate=RATE, seconds=0.4).frames(320))

        def cue():
            events.append("GO")
            time.sleep(0.25)

        started = time.perf_counter()
        stats = open_and_capture(
            source=source,
            pre=pre,
            segmenter=segmenter,
            ring=ring,
            quiet_window_s=0.1,
            limit_s=5.0,
            stop_at_segments=99,
            on_ready=cue,
        )
        elapsed = time.perf_counter() - started
        assert stats["wall_seconds"] < elapsed - 0.15, "the countdown must not be billed to the capture"
