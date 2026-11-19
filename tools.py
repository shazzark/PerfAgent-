"""Tools exposed to the PerfAgent orchestrator.

Three capabilities, one measurement model:

* `run_load_test` - latency/throughput only (Phase 1/2).
* `run_full_diagnostic` - the Phase 3 tool. It starts the load test on a
  background thread, samples machine CPU/memory with `profiler.get_system_metrics`
  in the foreground, and returns both halves as one JSON-serialisable dict so the
  model can correlate "slow" with "busy" instead of guessing.
* `find_breaking_point` - the capacity ramp. It offers an increasing request rate
  in one-second bursts and keeps climbing past partial failures and slow answers
  until a step fails every request or the target stops accepting connections.

All return plain data that can be handed straight back to the LLM as a `tool`
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

# Ramp-up policy. A step is one second of offered traffic at a constant rate.
# There is deliberately no failure-rate or latency threshold: partial failures
# and multi-second answers are measurements, not stop conditions. The ramp ends
# only when a step fails 100% of its requests or the target stops accepting
# connections.
RAMP_BURST_SECONDS = 1.0

# Default search ceiling for the ramp. A single Python process driving threaded
# requests is honest to roughly 1,000 offered req/s on a typical laptop; five
# figure rates (20,000+) need an event-driven or compiled generator such as k6
# or wrk. This is the ramp's search ceiling, not a capacity claim about the
# target.
DEFAULT_MAX_RPS = 1000

# Above this average wait before a worker picks a request up, the 10-worker
# client itself is the queue. The ramp still reports the limit, but the caller
# must not present it as pure server capacity.
RAMP_CLIENT_QUEUE_WARNING_SECONDS = 0.25

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

__all__ = ["run_load_test", "run_full_diagnostic", "find_breaking_point"]


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


def _ramp_request(
    url: str,
    submitted_at: float,
    timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
) -> tuple[bool, float, float, str | None]:
    """Return `(succeeded, latency, worker_wait, failure_kind)` for one GET.

    `failure_kind` is None for HTTP 200, `HTTP <status>` for a non-200 response,
    `connection_refused` / `connection_error` when the target cannot be reached,
    and `timeout` when the client deadline expires. The old boolean-only helper
    could not tell a saturated API (5xx) from a dead one (connection refused), and
    the ruthless ramp has to distinguish exactly those two outcomes.

    The third value is how long the request sat in the executor queue before a
    worker started it. That is the difference between measuring the API and
    measuring this agent's own worker ceiling.
    """
    started = time.perf_counter()
    try:
        response = requests.get(url, timeout=timeout_seconds)
        if response.status_code == 200:
            succeeded, failure_kind = True, None
        else:
            succeeded, failure_kind = False, f"HTTP {response.status_code}"
    except requests.exceptions.ConnectionError as exc:
        # `requests` wraps refused connections, DNS failures, and resets in this
        # exception. A refused socket is the clearest "the API is dead" signal,
        # so it gets its own label for the crash reason.
        message = str(exc).lower()
        succeeded = False
        failure_kind = (
            "connection_refused"
            if "refused" in message or "10061" in message
            else "connection_error"
        )
    except requests.exceptions.Timeout:
        succeeded, failure_kind = False, "timeout"
    except requests.exceptions.RequestException:
        succeeded, failure_kind = False, "request_error"

    finished = time.perf_counter()
    return succeeded, finished - started, max(0.0, started - submitted_at), failure_kind


def _summarize_ramp_burst(
    url: str,
    offered_rps: int,
    outcomes: list[tuple[bool, float, float, str | None]],
    elapsed: float,
) -> dict[str, Any]:
    """Turn one burst's `(success, latency, queue_wait, failure_kind)` tuples into metrics."""
    total_requests = len(outcomes)
    latencies = sorted(latency for _succeeded, latency, _wait, _kind in outcomes)
    queue_waits = [wait for _succeeded, _latency, wait, _kind in outcomes]
    successful = sum(1 for succeeded, _latency, _wait, _kind in outcomes if succeeded)
    failed = total_requests - successful
    failure_rate = failed / total_requests if total_requests else 0.0
    queue_avg = sum(queue_waits) / total_requests if total_requests else 0.0
    queue_max = max(queue_waits) if queue_waits else 0.0
    avg_latency = sum(latencies) / total_requests if total_requests else 0.0
    # End-to-end latency is what a client offering this rate actually waits:
    # API time plus time spent queued behind the generator's workers. The HTTP
    # latency stays separate so client queueing is never reported as server
    # think-time.
    end_to_end_avg = avg_latency + queue_avg
    # Nearest-rank p95, matching run_load_test.
    p95_index = min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)

    # Keep the failure modes separate: "100% HTTP 500s" and "connection refused"
    # are different crashes, and the agent reports which one ended the ramp.
    failure_breakdown: dict[str, int] = {}
    for succeeded, _latency, _wait, kind in outcomes:
        if not succeeded:
            key = kind or "error"
            failure_breakdown[key] = failure_breakdown.get(key, 0) + 1
    connection_failures = sum(
        count
        for kind, count in failure_breakdown.items()
        if kind.startswith("connection_")
    )

    return {
        "url": url,
        "offered_rps": offered_rps,
        "burst_seconds": RAMP_BURST_SECONDS,
        "total_requests": total_requests,
        "successful": successful,
        "failed": failed,
        "failure_rate": round(failure_rate, 4),
        "failure_rate_percent": round(failure_rate * 100, 2),
        "failure_breakdown": failure_breakdown,
        "connection_failures": connection_failures,
        "all_failed": total_requests > 0 and successful == 0,
        "total_time_seconds": round(elapsed, 2),
        "completion_rps": round(total_requests / elapsed, 2) if elapsed > 0 else 0.0,
        "avg_latency_seconds": round(avg_latency, 3),
        "p95_latency_seconds": round(latencies[p95_index], 3),
        "max_latency_seconds": round(latencies[-1], 3),
        "end_to_end_avg_latency_seconds": round(end_to_end_avg, 3),
        # A non-zero queue wait means the client had no free worker; it is not
        # server think-time and must not be reported as such.
        "client_queue_avg_seconds": round(queue_avg, 3),
        "client_queue_max_seconds": round(queue_max, 3),
        "client_concurrency_ceiling": MAX_WORKERS,
        "client_limited": bool(
            total_requests > MAX_WORKERS
            and queue_avg > RAMP_CLIENT_QUEUE_WARNING_SECONDS
        ),
    }


def _connection_crash_reason(exc: BaseException) -> str:
    """Name the connection failure that ended the ramp.

    Windows reports a closed port as WinError 10061 ("actively refused it") and
    POSIX as `Connection refused`; both mean the API is no longer listening. Any
    other connection error still means the target is unreachable, just without a
    refusal to quote.
    """
    message = str(exc).lower()
    if "refused" in message or "10061" in message:
        return "Connection Refused"
    return "Connection Error (API unreachable)"


def _total_failure_reason(metrics: dict[str, Any]) -> str:
    """Name the dominant failure mode of a step where nothing succeeded."""
    breakdown = metrics.get("failure_breakdown") or {}
    if len(breakdown) == 1:
        kind = next(iter(breakdown))
        if kind.startswith("HTTP "):
            # "HTTP 500" -> "100% HTTP 500s"
            return f"100% {kind}s"
        if kind == "timeout":
            return "100% timeouts"
        if kind.startswith("connection_"):
            return (
                "Connection Refused"
                if kind == "connection_refused"
                else "Connection Error (API unreachable)"
            )
    return "100% of requests failed"


def _unreachable_step_metrics(
    url: str, offered_rps: int, error_message: str
) -> dict[str, Any]:
    """Build the step payload for a burst that could not reach the target.

    When the client raises before a single request is measured there are no
    latency figures to report. The record still has to carry the rate and the
    error, because those are the crash evidence the caller reports.
    """
    return {
        "url": url,
        "offered_rps": offered_rps,
        "burst_seconds": RAMP_BURST_SECONDS,
        "total_requests": 0,
        "successful": 0,
        "failed": 0,
        "failure_rate": 0.0,
        "failure_rate_percent": 0.0,
        "failure_breakdown": {"connection_error": 1},
        "connection_failures": 1,
        "all_failed": True,
        "total_time_seconds": 0.0,
        "completion_rps": 0.0,
        "avg_latency_seconds": 0.0,
        "p95_latency_seconds": 0.0,
        "max_latency_seconds": 0.0,
        "end_to_end_avg_latency_seconds": 0.0,
        "client_queue_avg_seconds": 0.0,
        "client_queue_max_seconds": 0.0,
        "client_concurrency_ceiling": MAX_WORKERS,
        "client_limited": False,
        "error": error_message,
    }


def _run_ramp_step(url: str, offered_rps: int) -> dict[str, Any]:
    """Offer `offered_rps` requests over one second and measure the burst.

    Requests are paced `1 / rps` seconds apart so the step represents a request
    rate rather than a single concurrent clump. The executor still caps in-flight
    work at `MAX_WORKERS`, which is deliberate: `client_limited` then tells the
    caller whether the observed latency belongs to the API or to the generator.
    """
    total_requests = max(1, round(offered_rps * RAMP_BURST_SECONDS))
    interval = RAMP_BURST_SECONDS / total_requests
    print(
        f"Ramp step: offering {offered_rps} req/s "
        f"({total_requests} requests over {RAMP_BURST_SECONDS:.0f}s)..."
    )

    started = time.perf_counter()
    submissions: list[concurrent.futures.Future] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for index in range(total_requests):
            scheduled = started + index * interval
            wait = scheduled - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            submitted_at = time.perf_counter()
            submissions.append(
                executor.submit(_ramp_request, url, submitted_at)
            )
        outcomes = [future.result() for future in submissions]

    elapsed = time.perf_counter() - started
    result = _summarize_ramp_burst(url, offered_rps, outcomes, elapsed)
    print(f"  Step result: {result}")
    return result


def find_breaking_point(
    url: str,
    start_rps: int = 10,
    step_increase: int | None = None,
    max_rps: int = DEFAULT_MAX_RPS,
) -> dict[str, Any]:
    """Ramp offered load until the API is completely dead.

    Each step offers a constant `rps` for one second. A step that merely looks
    unwell does not end the ramp: partial failures and multi-second answers are
    recorded and the next step still runs, because the question is where the API
    stops serving, not where it first becomes uncomfortable.

    The ramp ends at the first step that either

    * fails 100% of its requests (for example every response is HTTP 500), or
    * reports a connection failure, i.e. the target no longer accepts connections.

    `step_increase=None` (the default) doubles the offered rate after every
    surviving step (10 -> 20 -> 40 -> 80), which reaches the crash point in a
    handful of steps. An explicit integer switches to additive steps
    (`start_rps + step_increase`, ...) when a finer table is wanted.

    `max_rps` defaults to 1000, roughly where a single Python process driving
    threaded requests stops being an honest load generator on a typical laptop.
    Reaching 20,000+ req/s needs an event-driven or compiled generator such as
    k6 or wrk; a larger `max_rps` here reports where this client ran out of
    offered rate, not where the server ran out of capacity.

    Returns:
        A JSON-serialisable dict with `breaking_rps` (the crashing rate, or None
        if the API survived `max_rps`), `last_successful_rps`, `crash_reason`
        (for example "100% HTTP 500s" or "Connection Refused"), `final_metrics`
        (the crashing step's metrics, or the last step when nothing crashed),
        every attempted step in `steps`, and the ramp strategy.

        `break_was_client_limited` means this agent's fixed worker pool was
        queueing during the crash step, so the reported rate is a lower bound on
        what the target could be driven to, not a hardware capacity figure.
        `breaking_reason` is kept as an alias of `crash_reason` for callers
        written before the ruthless ramp.
    """
    for name, value in (("start_rps", start_rps), ("max_rps", max_rps)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if step_increase is not None:
        if isinstance(step_increase, bool) or not isinstance(step_increase, int):
            raise ValueError("step_increase must be an integer or None")
        if step_increase < 1:
            raise ValueError("step_increase must be at least 1 when provided")
    if start_rps < 1:
        raise ValueError("start_rps must be at least 1")
    if max_rps < start_rps:
        raise ValueError("max_rps must be greater than or equal to start_rps")

    strategy = "doubling" if step_increase is None else f"+{step_increase} req/s"
    print(
        f"Ramping {url} from {start_rps} req/s up to at most {max_rps} req/s "
        f"({strategy}) until the API is completely dead..."
    )

    steps: list[dict[str, Any]] = []
    last_successful_rps: int | None = None
    crash_rps: int | None = None
    crash_reason: str | None = None
    final_metrics: dict[str, Any] | None = None
    offered_rps = start_rps

    while offered_rps <= max_rps:
        try:
            metrics = _run_ramp_step(url, offered_rps)
        except requests.exceptions.ConnectionError as exc:
            # A step can still raise before any request is measured (a mocked
            # client, DNS failure, or the port closing between submission and
            # connect). An unreachable target is exactly the crash this ramp is
            # looking for, so record it and stop.
            crash_rps = offered_rps
            crash_reason = _connection_crash_reason(exc)
            final_metrics = _unreachable_step_metrics(url, offered_rps, str(exc))
            steps.append(final_metrics)
            print(f"CRASH: API unreachable at {offered_rps} req/s: {crash_reason}")
            break

        steps.append(metrics)
        final_metrics = metrics

        # Condition (b): a refused or reset socket means the API stopped
        # accepting work. This outranks the failure-rate check because it names
        # the actual cause of death.
        if int(metrics.get("connection_failures", 0)) > 0:
            breakdown = metrics.get("failure_breakdown") or {}
            crash_rps = offered_rps
            crash_reason = (
                "Connection Refused"
                if "connection_refused" in breakdown
                else "Connection Error (API unreachable)"
            )
            print(f"CRASH: API unreachable at {offered_rps} req/s: {crash_reason}")
            break

        # Condition (a): every request in the step failed. Partial failures are
        # deliberately not enough; the API has to be completely dead.
        if int(metrics.get("failed", 0)) > 0 and int(metrics.get("successful", 0)) == 0:
            crash_rps = offered_rps
            crash_reason = _total_failure_reason(metrics)
            print(f"CRASH: API dead at {offered_rps} req/s: {crash_reason}")
            break

        last_successful_rps = offered_rps
        if offered_rps >= max_rps:
            break
        if step_increase is None:
            offered_rps = min(offered_rps * 2, max_rps)
        else:
            offered_rps = min(offered_rps + step_increase, max_rps)

    stop_conditions = {
        "stop_on_total_failure": True,
        "stop_on_connection_error": True,
        "failure_rate_percent": 100.0,
    }

    if crash_reason is not None:
        result = {
            "url": url,
            "start_rps": start_rps,
            "step_increase": step_increase,
            "max_rps": max_rps,
            "strategy": "double" if step_increase is None else "linear",
            "found": True,
            "breaking_rps": crash_rps,
            "last_successful_rps": last_successful_rps,
            "crash_reason": crash_reason,
            # Alias kept for callers written before the ruthless ramp; both
            # names describe the same stop event.
            "breaking_reason": crash_reason,
            "breaking_point": final_metrics,
            "final_metrics": final_metrics,
            "steps": steps,
            "stop_conditions": stop_conditions,
            "client_concurrency_ceiling": MAX_WORKERS,
            "break_was_client_limited": bool(
                final_metrics and final_metrics.get("client_limited")
            ),
        }
        print(
            f"BREAKING POINT: {crash_rps} req/s: {crash_reason} "
            f"(last surviving rate: {last_successful_rps})"
        )
        return result

    print(
        f"API survived {max_rps} req/s offered load without a total crash "
        f"(last surviving rate: {last_successful_rps})."
    )
    return {
        "url": url,
        "start_rps": start_rps,
        "step_increase": step_increase,
        "max_rps": max_rps,
        "strategy": "double" if step_increase is None else "linear",
        "found": False,
        "breaking_rps": None,
        "last_successful_rps": last_successful_rps,
        "crash_reason": None,
        # Alias of crash_reason; both are null when nothing crashed.
        "breaking_reason": None,
        "breaking_point": None,
        "final_metrics": final_metrics,
        "steps": steps,
        "stop_conditions": stop_conditions,
        "client_concurrency_ceiling": MAX_WORKERS,
        "break_was_client_limited": bool(
            final_metrics and final_metrics.get("client_limited")
        ),
    }


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
