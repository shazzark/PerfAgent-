"""Load-testing tools exposed to the PerfAgent orchestrator.

`run_load_test` is the only capability the Phase 2 agent can call. It fires a
burst of HTTP GET requests at a target URL through a thread pool and reports
latency, success/failure counts, and throughput as a JSON-serialisable dict,
so the result can be handed straight back to the LLM as a tool message.
"""

import concurrent.futures
import time

import requests

# Concurrency and timeout are fixed for Phase 2; they become tunable when the
# profiler (Phase 3) needs to drive different load shapes.
MAX_WORKERS = 10
REQUEST_TIMEOUT_SECONDS = 3.0

__all__ = ["run_load_test"]


def _request_succeeded(url: str) -> bool:
    """Return True only when a single GET receives HTTP 200.

    Connection errors, timeouts, and non-200 responses all count as failures,
    which matches the Phase 1 measurement semantics.
    """
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        return response.status_code == 200
    except requests.RequestException:
        # Timeouts and connection resets are expected under load; they are
        # data points for the report, not something the caller should handle.
        return False


def run_load_test(url: str, total_requests: int = 10) -> dict:
    """Send `total_requests` concurrent GET requests to `url` and measure them.

    Args:
        url: Absolute http(s) URL of the endpoint under test.
        total_requests: Number of requests to send. Must be at least 1.

    Returns:
        A JSON-serialisable summary::

            {
                "url": str,
                "total_requests": int,
                "total_time_seconds": float,
                "successful": int,
                "failed": int,
                "requests_per_second": float,
            }

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
            executor.map(lambda _: _request_succeeded(url), range(total_requests))
        )
    total_time = time.perf_counter() - start_time

    success_count = sum(outcomes)
    fail_count = total_requests - success_count
    requests_per_second = total_requests / total_time if total_time > 0 else 0.0

    result = {
        "url": url,
        "total_requests": total_requests,
        "total_time_seconds": round(total_time, 2),
        "successful": success_count,
        "failed": fail_count,
        "requests_per_second": round(requests_per_second, 2),
    }

    print(f"✅ Test complete: {result}")
    return result


if __name__ == "__main__":
    # Smoke test: start dummy_api first:
    #   py -m uvicorn dummy_api:app --reload
    run_load_test("http://127.0.0.1:8000/health", total_requests=10)
    run_load_test("http://127.0.0.1:8000/slow", total_requests=10)
