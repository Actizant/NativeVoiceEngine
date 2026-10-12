"""End-to-end: a full acceptance run with no approved model must report the gaps.

This is the file that proves the layer behaves as specified when everything is wired
together: transport latency, memory and CPU are *measured* on this machine, error
rates and speech synthesis claims are NOT_RUN naming the missing ingredient, the whole
run happens inside the offline guard, and the resulting verdict is INCOMPLETE rather
than a green board.
"""

from __future__ import annotations

import json
import math

from fixtures import silence, speech_like

from nativevoice.audio.pcm import measure
from nativevoice.audio.resample import resample
from nativevoice.core.language import Capability, CapabilityStatus, Task
from nativevoice.core.time import Timings
from nativevoice.diagnostics.offline import OfflineGuard
from nativevoice.diagnostics.timeline import Trace
from nativevoice.eval.capability_matrix import build_matrix
from nativevoice.eval.report import AcceptanceReport, CriterionStatus, Evidence, judge, not_run
from nativevoice.eval.resources import Sampler, read_cpu_time, read_memory
from nativevoice.eval.runner import Benchmark, Suite, run_benchmark
from nativevoice.eval.samples import builtin_fixtures
from nativevoice.eval.synthesis import intelligibility_proxy_check, silence_only


def _transport_work():
    buffer = speech_like()
    def work():
        converted = resample(buffer, 8000)
        return measure(converted).rms
    return work


def test_harness_measures_what_it_can_and_refuses_what_it_cannot(tmp_path):
    guard = OfflineGuard()
    trace = Trace()
    with guard:
        transport = Benchmark(name="resample-16k-to-8k", subject="audio.transport",
                              parameters={"frames": 14400}, repeats=5, warmup=2)
        result = run_benchmark(transport, _transport_work())
        sampler = Sampler(interval_s=0.005)
        payload, run = sampler.run(_transport_work())

        timings = Timings(started_ns=0, first_result_ns=1_000_000, finished_ns=9_000_000, audio_duration_s=0.0)
        trace.record("capture", timings, language="en", operation_id="e2e")

        rows = [result, run]

    assert result.status == "OK" and result.min_ms > 0
    assert run.measured and run.peak_bytes and run.peak_bytes > 0
    assert math.isfinite(payload)
    guard.assert_clean()

    report = AcceptanceReport("ULTRON Native Voice - local acceptance", host=rows[0].host)
    manifest_gaps = builtin_fixtures().gaps()
    evidence = Evidence.capture("python -m pytest tests/test_eval_end_to_end.py", before={"samples": manifest_gaps}, started_s=0.0)
    # The ceiling is a real-time factor, not an absolute millisecond figure: 0.9 s of
    # audio resampled in pure Python costs tens of milliseconds, and only the ratio to
    # the audio duration says whether a profile can sustain a live stream.
    audio_seconds = 14400 / 16000
    transport_rtf = rows[0].p95_ms / 1000 / audio_seconds
    report.add(judge("capture path p95 latency", rows[0].p95_ms, None, unit=" ms", evidence=evidence))
    report.add(judge("transport resample real-time factor", transport_rtf, 0.25, unit=" x realtime", evidence=evidence))
    report.add(judge("peak resident set size", read_memory().peak_mb, 4096.0, unit=" MB", direction="max"))
    report.add(judge("process cpu time accounted", read_cpu_time().total_s, 0.0, unit=" s", direction="min"))
    report.add(not_run(
        "word error rate per language",
        "no licensed, consented real-speech corpus is registered; builtin fixtures are synthetic and refused",
        threshold=0.15,
    ))
    report.add(not_run(
        "time-to-first-audio for synthesis",
        "no approved synthesis model installed, so there is no engine to time",
        unit=" ms", threshold=250.0,
    ))
    report.add(not_run("real-time factor for synthesis", "no approved synthesis model installed", unit="", threshold=1.0))

    summary = report.summarise()
    assert summary.failed == 0, report.render()
    assert summary.passed == 3
    assert summary.not_run == 4
    assert summary.verdict == "INCOMPLETE"
    assert "NOT_RUN" in report.render()

    payload_out = json.loads(json.dumps(report.to_dict()))
    assert payload_out["summary"]["not_run"] == 4
    by_name = {row["name"]: row for row in payload_out["criteria"]}
    # looked up by name, not position: a report is a set of criteria, and an
    # assertion tied to row order breaks the moment one is added.
    assert by_name["word error rate per language"]["measured"] is None
    assert by_name["word error rate per language"]["status"] == "NOT_RUN"
    assert by_name["transport resample real-time factor"]["status"] == "PASS"

    # The trace and the matrix agree that nothing was claimed about recognition.
    assert trace.stats("capture", "en") is not None
    assert trace.missing_stages("e2e") == ("vad", "stt", "routing", "tts", "playback")


def test_round_trip_intelligibility_is_refused_without_an_engine():
    check = intelligibility_proxy_check(silence(seconds=0.5), "the quick brown fox", "en")
    assert check.status is CriterionStatus.NOT_RUN
    assert "no transcriber supplied" in check.reason


def test_capability_matrix_from_builtin_fixtures_shows_nothing_measured():
    """An engine that only has synthetic fixtures available cannot claim MEASURED."""
    manifest = builtin_fixtures()
    capabilities = [
        Capability(
            task=Task.STT,
            status=CapabilityStatus.NO_MODEL,
            language=language,
            engine="shadow",
            limitations=("no usable corpus: every fixture is synthetic",),
        )
        for language in manifest.languages()
    ]
    matrix = build_matrix((c.language, c) for c in capabilities)
    verdict = matrix.verdict()
    assert verdict.measured == 0
    assert verdict.awaiting_model == len(manifest.languages())
    assert verdict.unmeasured == verdict.total
    assert "awaiting an approved model" in matrix.render_text()
    assert all(cell.limitations == ("no usable corpus: every fixture is synthetic",) for cell in matrix.cells)


def test_silence_only_output_is_flagged_and_never_reported_as_success():
    assert silence_only(silence(seconds=0.3)) is True
    assert silence_only(speech_like()) is False
