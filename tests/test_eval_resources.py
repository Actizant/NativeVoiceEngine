"""Resource measurement: it either reads the OS or says it could not."""

from __future__ import annotations

import math
import platform
import sys
import time

import pytest

from nativevoice.eval.resources import (
    Sampler,
    cpu_utilisation,
    read_cpu_time,
    read_memory,
)


def test_memory_reading_is_either_measured_or_explained():
    reading = read_memory()
    assert isinstance(reading.measured, bool)
    if reading.measured:
        assert reading.peak_bytes and reading.peak_bytes > 0
        assert reading.peak_mb > 0
        assert reading.source
        assert reading.reason == ""
    else:
        assert reading.peak_bytes is None
        assert reading.reason
        assert reading.available is False


def test_cpu_time_reading_is_either_measured_or_explained():
    reading = read_cpu_time()
    if reading.measured:
        assert reading.user_s is not None and reading.user_s >= 0
        assert reading.total_s >= reading.user_s
    else:
        assert reading.total_s is None and reading.reason


def test_windows_path_is_the_one_under_test_here():
    """Guard against silently falling back on the platform this suite ships on."""
    if platform.system() != "Windows":
        pytest.skip("windows-specific marshalling check")
    reading = read_memory()
    assert reading.measured, reading.reason
    assert reading.source.startswith("windows:")
    assert sys.maxsize > 2**32
    assert reading.peak_bytes > 0


def test_cpu_delta_grows_with_real_work():
    before = read_cpu_time()
    total = 0
    for value in range(400_000):
        total += value
    after = read_cpu_time()
    assert total == 400_000 * 399_999 // 2
    assert after.measured and before.measured
    assert after.total_s >= before.total_s


def test_utilisation_is_none_instead_of_zero_when_unmeasured():
    reading = read_cpu_time()
    assert cpu_utilisation(reading, reading, wall_s=1.0) == (0.0 if reading.measured else None)
    unmeasured = type(reading)(False, None, None, "test", "forced")
    assert cpu_utilisation(unmeasured, reading, wall_s=1.0) is None
    assert cpu_utilisation(reading, reading, wall_s=0.0) is None


def test_sampler_reports_peak_and_cpu_while_the_callable_runs():
    sampler = Sampler(interval_s=0.005)
    payload = []

    def workload() -> int:
        blocks = [bytearray(1024 * 512) for _ in range(8)]
        time.sleep(0.08)
        payload.append(blocks)
        return len(blocks)

    result, run = sampler.run(workload)
    assert result == 8
    assert run.polls >= 1, "the poller never ran; the sample set is empty, not zero"
    assert run.measured
    assert run.peak_bytes and run.peak_bytes > 0
    assert run.wall_s >= 0.05
    assert run.cpu_delta_s is not None and run.cpu_delta_s >= 0.0
    assert run.sampled
    assert run.reasons == ()


def test_sampler_teardown_happens_even_when_the_workload_raises():
    sampler = Sampler(interval_s=0.005)

    def boom():
        raise RuntimeError("workload exploded")

    with pytest.raises(RuntimeError):
        sampler.run(boom)
    assert not sampler._thread.is_alive()


def test_sampler_interval_must_be_positive():
    with pytest.raises(ValueError):
        Sampler(interval_s=0)


def test_run_serialises_for_a_report():
    _, run = Sampler(interval_s=0.01).run(lambda: sum(range(2000)))
    payload = run.as_dict()
    assert set(payload) >= {"measured", "polls", "peak_mb", "cpu_delta_s", "cpu_utilisation", "reasons"}
    assert payload["peak_mb"] is None or math.isfinite(payload["peak_mb"])
