"""Tools exposed to the PerfAgent orchestrator.

Two capabilities, one measurement model:

* `run_load_test` - latency/throughput only (Phase 1/2).
* `run_full_diagnostic` - the Phase 3 tool. It starts the load test on a
  background thread, samples machine CPU/memory with `profiler.get_system_metrics`
  in the foreground, and returns both halves as one JSON-serialisable dict so the
  model can correlate "slow" with "busy" instead of guessing.

Both return plain data that can be handed straight back to the LLM as a `tool`
message. HTTP-level failures are measurements, not exceptions.
"""

import concurrent.futures
import math
import threading
import time
from typing import Any

import requests

from profiler import get_system_metrics

# Concurrency is fixed for Phase 3; it becomes an input when the profiler needs
# to drive different load shapes.
MAX_WORKERS = 10
REQUEST_TIMEOUT_SECONDS = 3.0

# A CPU-bound target legitimately takes longer per request once concurrency
# oversubscribes the cores: /heavy time-boxes each request to 1.5 s of CPU, so a
# box with fewer logical CPUs than in-flight workers stretches that to several
# seconds of wall time. Diagnosing with the 3 s interactive deadline would then
# report real server latency as a client-side failure and hide the very signal
# being profiled, so the diagnostic path gets a wider deadline. Latency stays
# visible through wall time and requests/second.
DIAGNOSTIC_TIMEOUT_SECONDS = 15.0

# Default sampling window for a diagnostic. Long enough to cover the dummy API's
# 1.5 s per-request work several times over, short enough to keep an agent turn
# responsive.
PROFILE_WINDOW_SECONDS = 5

# A load-window reading alone cannot say whether the target caused the CPU it
# reports: this machine idles at 20-35% from unrelated applications, so an average
# of 36% is indistinguishable from sitting idle, and a first live run concluded
# "not CPU-bound, memory pressure" for an endpoint that peaked at 93%. A short
# idle sample taken immediately before the load test gives the payload a reference
# point, which is what turns a number into a correlation: "CPU rose from 24% to a
# 93% peak while the requests were in flight".
BASELINE_WINDOW_SECONDS = 2

__all__ = ["run_load_test", "run_full_diagnostic"]


def _request_succeeded(
    url: str, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS
) -> bool:
    """Return True only when a single GET receives HTTP 200.

    Connection errors, timeouts, and non-200 responses all count as failures,
    which matches the Phase 1 measurement semantics.
    """
    try:
        response = requests.get(url, timeout=timeout_seconds)
        return response.status_code == 200
    except requests.RequestException:
        # Timeouts and connection resets are expected under load; they are
        # data points for the report, not something the caller should handle.
        return False


def _timed_request(
    url: str, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS
) -> tuple[bool, float]:
    """Return `(succeeded, seconds)` for one GET.

    Timing wraps the request itself, so the latency figure is what a client
    actually waited rather than an average inferred from the whole burst.
    """
    started = time.perf_counter()
    succeeded = _request_succeeded(url, timeout_seconds)
    return succeeded, time.perf_counter() - started


def run_load_test(
    url: str,
    total_requests: int = 10,
    timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
) -> dict:
    """Send `total_requests` concurrent GET requests to `url` and measure them.

    Args:
        url: Absolute http(s) URL of the endpoint under test.
        total_requests: Number of requests to send. Must be at least 1.
        timeout_seconds: Per-request client deadline. Defaults to the interactive
            3 s budget; `run_full_diagnostic` passes a wider one.

    Returns:
        A JSON-serialisable summary::

            {
                "url": str,
                "total_requests": int,
                "total_time_seconds": float,
                "successful": int,
                "failed": int,
                "requests_per_second": float,
                "avg_latency_seconds": float,
                "p95_latency_seconds": float,
                "max_latency_seconds": float,
            }

        Latency percentiles cover every attempt, including timeouts, and are the
        only correct per-request figure: because requests run concurrently,
        `total_time_seconds / total_requests` understates what a single client
        waited. The p95 uses nearest-rank, so with 10 requests it is the slowest
        one.

    Raises:
        ValueError: If `total_requests` is not a positive integer.
    """
    if total_requests < 1:
        raise ValueError("total_requests must be at least 1")

    print(f"🚀 Starting load test on {url} with {total_requests} requests...")

    start_time = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Collect outcomes instead of mutating shared counters from worker
        # threads: no lock needed and the totals are exact.
        outcomes = list(
            executor.map(
                lambda _: _timed_request(url, timeout_seconds),
                range(total_requests),
            )
        )
    total_time = time.perf_counter() - start_time

    latencies = sorted(latency for _succeeded, latency in outcomes)
    success_count = sum(1 for succeeded, _latency in outcomes if succeeded)
    fail_count = total_requests - success_count
    requests_per_second = total_requests / total_time if total_time > 0 else 0.0
    # Nearest-rank p95: with a 10-request test this is the slowest attempt, which
    # is the number a user actually felt.
    p95_index = min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)

    result = {
        "url": url,
        "total_requests": total_requests,
        "total_time_seconds": round(total_time, 2),
        "successful": success_count,
        "failed": fail_count,
        "requests_per_second": round(requests_per_second, 2),
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 3),
        "p95_latency_seconds": round(latencies[p95_index], 3),
        "max_latency_seconds": round(latencies[-1], 3),
    }

    print(f"✅ Test complete: {result}")
    return result


def run_full_diagnostic(
    url: str,
    total_requests: int = 10,
    profile_seconds: int = PROFILE_WINDOW_SECONDS,
    baseline_seconds: int = BASELINE_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Load-test `url` while sampling CPU and memory on this machine.

    The load test runs on a background thread so the profiler can sample the
    machine at the same time the traffic is in flight. A short baseline is
    sampled first, before any traffic, so the CPU and memory figures have
    something to be compared against. The halves are merged into one payload
    because that is the unit of evidence the LLM reasons over: "N requests took T
    seconds *and* average CPU rose from B to C while they were in flight".

    Args:
        url: Absolute http(s) URL of the endpoint under test.
        total_requests: Number of concurrent-ish requests to send. Minimum 1.
        profile_seconds: Length of the CPU/memory sampling window. Minimum 1.
        baseline_seconds: Length of the pre-load idle sample. Minimum 1.

    Returns:
        A JSON-serialisable dict::

            {
                "url": str,
                "total_requests": int,
                "baseline": {                      # profiler.get_system_metrics
                    ...                            # sampled before the load test
                },
                "load_test": {                     # tools.run_load_test output
                    "url": str,
                    "total_requests": int,
                    "total_time_seconds": float,
                    "successful": int,
                    "failed": int,
                    "requests_per_second": float,
                    "avg_latency_seconds": float,
                    "p95_latency_seconds": float,
                    "max_latency_seconds": float,
                },
                "system_metrics": {                # profiler.get_system_metrics
                    ...                            # sampled during the load test
                },
                "cpu_delta_percent": float,        # load avg minus baseline avg
                "memory_delta_percent": float,     # load avg minus baseline avg
                "profile_covers_full_load_test": bool,
            }

        If the load worker fails unexpectedly, `load_test` is `None` and `error`
        explains why. `profile_covers_full_load_test` is False when the load test
        outlasted the sampling window, which means the metrics describe only the
        profiled prefix of the run.

    Raises:
        ValueError: If any of the window/count arguments is not positive.
    """
    if total_requests < 1:
        raise ValueError("total_requests must be at least 1")
    if profile_seconds < 1:
        raise ValueError("profile_seconds must be at least 1")
    if baseline_seconds < 1:
        raise ValueError("baseline_seconds must be at least 1")

    print(
        f"🔬 Full diagnostic on {url}: {total_requests} requests while profiling "
        f"{profile_seconds}s of CPU/memory..."
    )
    print(f"🌡️  Sampling a {baseline_seconds}s idle baseline before the load test...")
    baseline = get_system_metrics(duration_seconds=baseline_seconds)

    # Shared state is a plain dict written once by the worker and read only after
    # join(), so no lock is needed.
    outcome: dict[str, Any] = {}

    def _load_worker() -> None:
        try:
            outcome["load_test"] = run_load_test(
                url=url,
                total_requests=total_requests,
                timeout_seconds=DIAGNOSTIC_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - must not vanish inside a thread
            # An exception raised in a thread would be invisible to the caller and
            # the diagnostic would return a half-empty payload with no reason.
            outcome["error"] = f"{type(exc).__name__}: {exc}"

    load_thread = threading.Thread(
        target=_load_worker, name="perfagent-load-test", daemon=True
    )
    load_thread.start()
    try:
        system_metrics = get_system_metrics(duration_seconds=profile_seconds)
    finally:
        # Always settle the load test before reporting, even if profiling failed:
        # returning while requests are still in flight would attribute unrelated
        # CPU to the next run.
        load_thread.join()

    load_test = outcome.get("load_test")
    result: dict[str, Any] = {
        "url": url,
        "total_requests": total_requests,
        "baseline": baseline,
        "system_metrics": system_metrics,
    }

    if load_test is None:
        result["load_test"] = None
        result["error"] = outcome.get("error", "load test produced no result")
        print(f"⚠️  Diagnostic incomplete: {result['error']}")
        return result

    result["load_test"] = load_test
    if "error" in outcome:
        result["error"] = outcome["error"]
    # Deltas are what make the reading interpretable: the absolute figures include
    # whatever else this machine is doing, the deltas are the change the load test
    # is responsible for.
    result["cpu_delta_percent"] = round(
        system_metrics["avg_cpu_percent"] - baseline["avg_cpu_percent"], 2
    )
    result["memory_delta_percent"] = round(
        system_metrics["avg_memory_percent"] - baseline["avg_memory_percent"], 2
    )
    # CPU is only meaningful for the part of the test inside the sampling window.
    result["profile_covers_full_load_test"] = (
        load_test["total_time_seconds"] <= system_metrics["duration_seconds"]
    )

    print(f"✅ Diagnostic complete: {result}")
    return result


if __name__ == "__main__":
    # Smoke test: start dummy_api first:
    #   py -m uvicorn dummy_api:app --reload
    run_load_test("http://127.0.0.1:8000/health", total_requests=10)
    run_load_test("http://127.0.0.1:8000/slow", total_requests=10)
    run_full_diagnostic("http://127.0.0.1:8000/heavy", total_requests=12)
