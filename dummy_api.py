"""Dummy target API used by PerfAgent as the system under test.

Each endpoint has a different bottleneck shape so the agent can learn to tell
them apart from the outside:

* ``/health`` - instant and CPU-light; the control.
* ``/slow``   - ``time.sleep``: high latency with (almost) no CPU cost, i.e. an
  I/O-bound or blocked handler.
* ``/heavy``  - a real CPU-bound hashing loop. Its latency is comparable to
  ``/slow`` but it keeps every logical CPU busy, so the profiler sees high CPU.

The contrast between ``/slow`` and ``/heavy`` is the point: same wall-clock
latency, completely different resource signature.

To run this: py -m uvicorn dummy_api:app --reload
"""

import hashlib
import time

from fastapi import FastAPI

app = FastAPI()

# How long /heavy should burn CPU per request. Matched to /slow's 1.5 s sleep so
# the two endpoints are comparable on latency and differ only in CPU cost.
CPU_BURN_SECONDS = 1.5

# Hashing a tiny digest keeps the GIL held, so concurrent handler threads
# serialize on one core: 8 concurrent requests measured 0.99 cores busy, i.e. the
# profiler saw ~12% CPU on an 8-core box while latency doubled from GIL
# contention. Hashing a block well above CPython's ~2 KB GIL-release threshold
# lets OpenSSL hash without the GIL, so concurrent requests really do spread
# across cores (8 threads -> 5.3 cores busy, latency still 1.5 s). That
# multi-core signature is what /heavy exists to give the profiler.
HASH_BLOCK_BYTES = 256 * 1024
HASH_BLOCK = b"perfagent-heavy-payload:" + bytes(range(256)) * (HASH_BLOCK_BYTES // 256)

# Hash a few blocks between clock reads: one 256 KB hash is well under a
# millisecond, so this keeps the burn on its `seconds` budget without spending
# measurable time in time.perf_counter().
HASHES_PER_CLOCK_CHECK = 4

# 1. A fast, healthy endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "I am fast and healthy!"}

# 2. A slow, bottlenecked endpoint (We will use this to test the bot)
@app.get("/slow")
def slow_endpoint():
    # Simulating a slow database query or heavy computation
    time.sleep(1.5) 
    return {"status": "ok", "message": "Wow, that took a long time."}


def _burn_cpu(seconds: float) -> tuple[int, float]:
    """Hash large blocks in a tight loop until `seconds` have elapsed.

    Wall-clock bounded rather than iteration bounded, so the endpoint costs
    roughly the same time on a fast laptop and a slow CI runner. Each hash is
    over `HASH_BLOCK`, which is deliberately larger than the GIL-release
    threshold so concurrent requests actually run in parallel. Returns the number
    of hashes computed and the elapsed time, which makes it obvious that the time
    was spent computing rather than waiting.
    """
    iterations = 0
    started = time.perf_counter()
    while True:
        # Check the clock once per chunk so the loop is not dominated by
        # time.perf_counter() calls.
        for _ in range(HASHES_PER_CLOCK_CHECK):
            hashlib.sha256(HASH_BLOCK).digest()
        iterations += HASHES_PER_CLOCK_CHECK
        if time.perf_counter() - started >= seconds:
            break
    return iterations, time.perf_counter() - started


# 3. A CPU-bound endpoint: same latency ballpark as /slow, but it actually burns
#    CPU. Deliberately a sync `def` so FastAPI runs it in the worker threadpool,
#    where concurrent requests spread across cores just like a real CPU-bound
#    Python service does.
@app.get("/heavy")
def heavy_endpoint():
    iterations, elapsed = _burn_cpu(CPU_BURN_SECONDS)
    return {
        "status": "ok",
        "message": "That burned real CPU cycles.",
        "iterations": iterations,
        "compute_seconds": round(elapsed, 3),
    }


# To run this: py -m uvicorn dummy_api:app --reload
