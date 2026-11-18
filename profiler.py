"""Machine-level resource profiler - Phase 3: The Eyes.

`tools.run_load_test` answers "how slow was it?". This module answers "why?",
by sampling CPU and memory on the machine that is actually serving the traffic.

The sampler is deliberately simple: one `psutil.cpu_percent(interval=1)` reading
per second. `psutil` measures CPU utilisation across an interval, so the call
blocks for that interval and returns an accurate figure for it; calling
`cpu_percent()` without an interval returns 0.0 on the first call because there
is no previous sample to compare against.

Scope and caveats:

* Metrics are **machine-wide**, not per-process. When the load test runs in the
  same process as the profiler (the default), the load generator's own CPU is
  included in the reading. That is acceptable for "is this workload CPU-bound?"
  and must not be read as an exact server-side CPU number.
* `avg_memory_percent` is system-wide memory utilisation from
  `psutil.virtual_memory()`; it moves slowly and rarely identifies a bottleneck
  in a short CPU-bound test.

Usage:
    py profiler.py
"""

import time
from typing import Any

import psutil

__all__ = ["get_system_metrics"]

DEFAULT_DURATION_SECONDS = 5
SAMPLE_INTERVAL_SECONDS = 1.0


def get_system_metrics(duration_seconds: int = DEFAULT_DURATION_SECONDS) -> dict[str, Any]:
    """Measure average CPU and memory utilisation over `duration_seconds`.

    Args:
        duration_seconds: Number of one-second samples to take. Must be >= 1.

    Returns:
        A JSON-serialisable dict::

            {
                "avg_cpu_percent": float,      # mean CPU over the window, 0-100
                "avg_memory_percent": float,   # mean RAM in use, 0-100
                "peak_cpu_percent": float,     # highest single-second reading
                "peak_memory_percent": float,
                "cpu_samples": list[float],    # per-second CPU readings
                "memory_samples": list[float], # per-second RAM readings
                "samples": int,                # readings actually taken
                "duration_seconds": float,     # measured window length
                "logical_cpu_count": int,      # context for the CPU percentage
            }

        `avg_cpu_percent` is normalised across all cores: 100 means every logical
        CPU was busy for the whole window, so a single saturated core on an
        8-core machine reads around 12.5.

        The per-second series is included because the summary hides shape: a
        window averaging 40% can be a steady 40% or 95% for the first half and
        idle after the load finished, and only the series tells those apart.

    Raises:
        ValueError: If `duration_seconds` is not a positive integer.
    """
    if not isinstance(duration_seconds, int) or duration_seconds < 1:
        raise ValueError("duration_seconds must be a positive integer")

    cpu_readings: list[float] = []
    memory_readings: list[float] = []

    start = time.perf_counter()
    for _ in range(duration_seconds):
        # Blocks for SAMPLE_INTERVAL_SECONDS and returns utilisation over exactly
        # that interval - the accurate way to use psutil.cpu_percent.
        cpu_readings.append(psutil.cpu_percent(interval=SAMPLE_INTERVAL_SECONDS))
        memory_readings.append(psutil.virtual_memory().percent)
    elapsed = time.perf_counter() - start

    samples = len(cpu_readings)
    return {
        "avg_cpu_percent": round(sum(cpu_readings) / samples, 2),
        "avg_memory_percent": round(sum(memory_readings) / samples, 2),
        "peak_cpu_percent": round(max(cpu_readings), 2),
        "peak_memory_percent": round(max(memory_readings), 2),
        "cpu_samples": [round(value, 2) for value in cpu_readings],
        "memory_samples": [round(value, 2) for value in memory_readings],
        "samples": samples,
        "duration_seconds": round(elapsed, 2),
        "logical_cpu_count": psutil.cpu_count(logical=True) or 1,
    }


if __name__ == "__main__":
    # Smoke test: idle machine metrics, no load required.
    print(f"🔬 Sampling system metrics for {DEFAULT_DURATION_SECONDS}s...")
    print(get_system_metrics())
