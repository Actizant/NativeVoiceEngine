"""Benchmark harness, baseline regression, and suite fault isolation."""

from __future__ import annotations

import json
import math
import random

import pytest

from nativevoice.core.errors import CapabilityUnavailable
from nativevoice.eval.runner import (
    Benchmark,
    BenchmarkResult,
    check_regression,
    load_baseline,
    percentile,
    run_benchmark,
    save_baseline,
    Suite,
    compare,
)


def _busy(iterations: int = 20000):
    def work() -> int:
        total = 0
        for value in range(iterations):
            total += value
        return total

    return work


def test_percentile_is_nearest_rank_and_never_interpolated():
    # Nearest-rank: the p95 of 20 sorted samples is the 19th value, a rank that
    # actually occurred. Interpolation would report a time no run ever took.
    values = [float(v) for v in range(1, 21)]
    assert percentile(values, 0.95) == 19.0
    assert percentile(values, 1.0) == 20.0
    assert percentile(values, 0.5) == 10.0
    assert percentile([5.0], 0.95) == 5.0
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile(values, 0.0)


def test_benchmark_produces_min_median_p95_max_and_every_sample():
    result = run_benchmark(Benchmark(name="sum", subject="cpu", repeats=7, warmup=2), _busy())
    assert result.status == "OK"
    assert len(result.samples_ms) == 7
    assert result.min_ms <= result.median_ms <= result.p95_ms <= result.max_ms
    assert result.repeats == 7 and result.warmup == 2
    assert math.isfinite(result.mean_ms) and result.mean_ms >= 0
    assert result.host and result.measured_at


def test_callable_shortcut_needs_no_descriptor():
    result = run_benchmark(_busy(1000), repeats=3)
    assert result.name == "_busy" or result.name
    assert len(result.samples_ms) == 3


def test_benchmark_without_a_callable_is_refused():
    with pytest.raises(CapabilityUnavailable):
        run_benchmark(Benchmark(name="nothing"))


def test_seeded_benchmark_passes_a_deterministic_rng():
    descriptor = Benchmark(name="rng-work", subject="sampled", repeats=3, warmup=0, seed=1234)

    def work(rng: random.Random) -> None:
        work.values = getattr(work, "values", []) + [rng.random()]

    first = run_benchmark(descriptor, work)
    assert first.seed == 1234
    assert len(getattr(work, "values", [])) == 3


def test_result_json_round_trip_is_lossless_enough_for_comparison(tmp_path):
    result = run_benchmark(Benchmark(name="rt", subject="s", parameters={"rate": 16000}, repeats=4), _busy(500))
    path = str(tmp_path / "baseline.json")
    save_baseline(path, result)
    loaded = load_baseline(path)
    assert loaded.name == result.name
    assert loaded.parameters == {"rate": 16000}
    # Stored values are rounded to 4 dp of a millisecond (0.1 us), so the round trip is
    # exact to within half a rounding step - an absolute bound, since a sub-microsecond
    # benchmark would make any relative bound meaningless.
    assert abs(loaded.median_ms - result.median_ms) <= 5e-4
    assert abs(loaded.p95_ms - result.p95_ms) <= 5e-4
    assert loaded.samples_ms == tuple(round(v, 4) for v in result.samples_ms)


def test_first_run_reports_no_baseline_rather_than_improvement(tmp_path):
    path = str(tmp_path / "nested" / "baseline.json")
    descriptor = Benchmark(name="latency", subject="s", repeats=3)
    result, outcome = check_regression(descriptor, _busy(500), path)
    assert outcome.kind == "no_baseline"
    assert outcome.accept is None
    assert outcome.has_baseline is False
    assert "no stored baseline" in outcome.detail
    import os

    assert os.path.exists(path), "the first run must record the baseline it could not compare to"


def test_corrupt_baseline_is_an_error_not_a_silent_no_baseline(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CapabilityUnavailable):
        load_baseline(str(path))


def test_regression_direction_is_detected(tmp_path):
    baseline = run_benchmark(Benchmark(name="x", subject="s", repeats=3), _busy(1000))
    faster = run_benchmark(Benchmark(name="x", subject="s", repeats=3), _busy(10))
    slower = run_benchmark(Benchmark(name="x", subject="s", repeats=3), _busy(400_000))
    assert compare(faster, baseline, tolerance_pct=5).kind in ("improved", "unchanged")
    regressed = compare(slower, baseline, tolerance_pct=5)
    assert regressed.kind == "regressed"
    assert regressed.accept is False
    assert regressed.delta_pct > 5


def test_mismatched_parameter_sets_are_not_compared():
    baseline = run_benchmark(Benchmark(name="x", subject="stt", parameters={"rate": 16000}, repeats=2), _busy(100))
    current = run_benchmark(Benchmark(name="x", subject="stt", parameters={"rate": 44100}, repeats=2), _busy(100))
    outcome = compare(current, baseline)
    assert outcome.kind == "incompatible"
    assert outcome.delta_pct is None
    assert outcome.accept is None


def test_zero_baseline_median_is_declared_unmeasurable():
    baseline = BenchmarkResult("x", "s", {}, 2, 0, None, 0.0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0))
    current = run_benchmark(Benchmark(name="x", subject="s", repeats=2), _busy(100))
    assert compare(current, baseline).kind == "unmeasurable"


def test_suite_isolates_a_crashing_benchmark_and_keeps_the_others():
    def broken():
        raise ValueError("engine exploded")

    suite = Suite("acceptance")
    suite.add(Benchmark(name="good-one", subject="s", repeats=2), _busy(1000))
    suite.add(Benchmark(name="bad-one", subject="s", repeats=2), broken)
    suite.add(Benchmark(name="skipped", subject="s", repeats=2), _busy(1000), skip_reason="no approved model for bn")
    suite.add(Benchmark(name="no-callable", subject="s", repeats=2), None)
    suite.add(Benchmark(name="good-two", subject="s", repeats=2), _busy(1000))
    suite.run()

    counts = suite.counts()
    assert counts == {"OK": 2, "ERROR": 1, "NOT_RUN": 2}
    rows = {row.name: row for row in suite.rows}
    assert rows["bad-one"].status == "ERROR"
    assert "engine exploded" in rows["bad-one"].reason
    assert rows["bad-one"].result is None
    assert rows["skipped"].reason == "no approved model for bn"
    assert "without a callable" in rows["no-callable"].reason
    assert rows["good-one"].result is not None
    rendered = suite.render_text()
    assert "ERROR" in rendered and "NOT_RUN" in rendered
    payload = json.loads(json.dumps(suite.to_dict()))
    assert payload["counts"]["ERROR"] == 1
    assert payload["rows"][0]["result"]["name"] == "good-one"


def test_suite_records_regression_rows_when_a_baseline_path_is_given(tmp_path):
    path = str(tmp_path / "b.json")
    suite = Suite("with-baseline")
    suite.add(Benchmark(name="lat", subject="s", repeats=3), _busy(200_000), baseline_path=path, tolerance_pct=50.0)
    suite.run()
    assert suite.rows[0].outcome.kind == "no_baseline"
    suite.run()
    # The point of the second run is that a comparison could be made at all, and that
    # the stored baseline is not silently replaced by whatever today's run measured.
    assert suite.rows[0].outcome.has_baseline is True
    assert suite.rows[0].outcome.kind in ("unchanged", "improved", "regressed")
    assert suite.rows[0].outcome.baseline.repeats == 3
    assert load_baseline(path).measured_at == suite.rows[0].outcome.baseline.measured_at


def test_benchmark_descriptor_validates_its_counts():
    with pytest.raises(ValueError):
        Benchmark(name="x", repeats=0)
    with pytest.raises(ValueError):
        Benchmark(name="x", warmup=-1)
