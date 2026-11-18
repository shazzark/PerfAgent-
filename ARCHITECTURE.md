# Architecture

PerfAgent is a small modular monolith: one Python process talks to an LLM and
executes local load tests. There is no framework layer between the orchestrator
and the tool — the OpenAI-compatible tool-calling contract *is* the interface.

## Stack

| Layer | Technology | Role |
| --- | --- | --- |
| Language | Python 3.13 | Runtime for agent, tool, and target API |
| LLM access | OpenAI Python SDK (`openai`) | Chat completions + function/tool calling |
| LLM provider | Groq (`https://api.groq.com/openai/v1`) | OpenAI-compatible inference endpoint |
| Model | `openai/gpt-oss-120b` | Planning, tool selection, result analysis |
| Configuration | `python-dotenv` | Loads `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` from `.env` |
| Target API | FastAPI + Uvicorn | Local dummy service used as the system under test |
| HTTP client | Requests | Per-request timing and status inspection |
| Concurrency | `concurrent.futures.ThreadPoolExecutor` | 10 parallel in-flight requests |
| Parallelism | `threading.Thread` | Load test on a background thread beside the profiler |
| Resource metrics | `psutil` | Machine CPU and memory sampling (`profiler.py`) |
| Timing | `time.perf_counter()` | Monotonic wall-clock measurement |

## Components

```
agent.py            Orchestrator: prompt -> LLM -> tool call -> LLM -> answer
tools.py            run_load_test(url, total_requests) -> load metrics
                    run_full_diagnostic(url, total_requests) -> load + system metrics
profiler.py         get_system_metrics(duration_seconds) -> avg CPU/memory
load_test_tool.py   Deprecated re-export of run_load_test (Phase 1 compatibility)
dummy_api.py        FastAPI target: /health (instant), /slow (1.5 s sleep),
                    /heavy (1.5 s of real CPU-bound hashing)
list_models.py      Diagnostics: prints models available to the Groq key
test_connection.py  Diagnostics: single chat completion against the configured model
```

Each file has one job. `agent.py` owns conversation state and LLM I/O;
`tools.py` owns measurement orchestration; `profiler.py` owns resource sampling;
neither tool module knows the LLM exists.


## Request flow

```
User prompt
   │
   ▼
agent.run_agent()
   ├─ system prompt + user prompt + TOOLS schema  ──►  Groq chat.completions
   │                                                        │
   │                                        tool_calls? ────┴──► no ──► final answer
   │                                             │ yes
   │                                             ▼
   │                                  _execute_tool(name, args)
   │                                             │
   │                                  tools.run_full_diagnostic(...)
   │                                     ├─ Thread: run_load_test(requests)
   │                                     └─ Main:   profiler.get_system_metrics()
   │                                             │
   │                                  combined dict (JSON)
   │                                             │
   │             messages.append(assistant message)
   │             messages.append({"role": "tool", ...})
   │                                             │
   └──────────────────── loop (max 5 rounds) ◄───┘
```


The assistant message containing `tool_calls` is appended before its tool
results. OpenAI-compatible APIs reject a `tool` message whose `tool_call_id`
does not belong to a preceding assistant turn.

## Tool contract

Schema advertised to the model:

```json
{
  "name": "run_full_diagnostic",
  "parameters": {
    "type": "object",
    "properties": {
      "url": { "type": "string" },
      "total_requests": { "type": "integer", "minimum": 1 }
    },
    "required": ["url", "total_requests"]
  }
}
```

Returned payload (also the string content of the `tool` message):

```json
{
  "url": "http://127.0.0.1:8000/heavy",
  "total_requests": 12,
  "baseline": {
    "avg_cpu_percent": 37.9,
    "avg_memory_percent": 97.85,
    "peak_cpu_percent": 39.5,
    "peak_memory_percent": 99.2,
    "cpu_samples": [36.3, 39.5],
    "memory_samples": [96.5, 99.2],
    "samples": 2,
    "duration_seconds": 2.0,
    "logical_cpu_count": 8
  },
  "system_metrics": {
    "avg_cpu_percent": 61.6,
    "avg_memory_percent": 97.54,
    "peak_cpu_percent": 96.4,
    "peak_memory_percent": 98.3,
    "cpu_samples": [96.4, 76.2, 54.1, 49.9, 31.4],
    "memory_samples": [98.0, 98.3, 96.8, 97.2, 97.4],
    "samples": 5,
    "duration_seconds": 5.13,
    "logical_cpu_count": 8
  },
  "load_test": {
    "url": "http://127.0.0.1:8000/heavy",
    "total_requests": 12,
    "total_time_seconds": 3.12,
    "successful": 12,
    "failed": 0,
    "requests_per_second": 3.84,
    "avg_latency_seconds": 1.591,
    "p95_latency_seconds": 1.613,
    "max_latency_seconds": 1.613
  },
  "cpu_delta_percent": 23.7,
  "memory_delta_percent": -0.31,
  "profile_covers_full_load_test": true
}
```

Per-request latency lives in `avg_latency_seconds`, `p95_latency_seconds` and
`max_latency_seconds`, measured around each individual request. Dividing
`total_time_seconds` by `total_requests` is **not** a latency figure: with ten
workers in flight, 12 requests that each take ~1.6 s finish in ~3.1 s, so that
division reports 260 ms instead of the 1.6 s a client actually waited. The
percentiles cover every attempt (including timeouts, which then show up as the
client deadline) and p95 uses nearest-rank, so in a 10-request test it is simply
the slowest attempt.

A success is an HTTP `200` response inside the client deadline. The interactive
default is 3 seconds; `run_full_diagnostic` passes
`DIAGNOSTIC_TIMEOUT_SECONDS = 15` because a CPU-bound handler legitimately takes
longer per request once concurrency oversubscribes the cores, and counting that
latency as a client-side failure would hide the signal being profiled. Timeouts,
connection errors, and other status codes still count as failures and do **not**
raise, because under load those are measurements, not exceptions. Invalid
arguments are returned as `{"error": "..."}` so the model can self-correct.


`run_load_test(url, total_requests)` remains available for latency-only runs
and for the deprecated `load_test_tool.py` compatibility module; the agent only
advertises `run_full_diagnostic`.

## Profiler (`profiler.py`)

`get_system_metrics(duration_seconds=5)` takes one
`psutil.cpu_percent(interval=1)` reading per second and one
`psutil.virtual_memory().percent` reading alongside it, then returns the mean
and peak of each series plus the sample count, measured window length, and
logical CPU count.

Design points:

- **Interval-based sampling.** `psutil.cpu_percent()` without an interval
  returns `0.0` on its first call because it has no previous sample to diff
  against. Passing `interval=1` blocks for a second and returns utilisation over
  exactly that second, which is what makes the average trustworthy.
- **Machine-wide, not per-process.** Readings cover the whole host, including
  the load generator and any other process, so they answer "is this workload
  CPU-bound?" rather than "how much CPU did the server process use?".
  Process-scoped sampling (`psutil.Process(pid)`) is future work.
- **Coverage is reported, not assumed.** The sampling window is fixed at 5 s by
  default. When a load test runs longer, `profile_covers_full_load_test` is
  `false` and the CPU figure describes only the profiled prefix of the run.
- **The load thread is always joined.** If profiling raises, the `finally` block
  still waits for in-flight requests so the next run's CPU sample is not
  polluted. Exceptions inside the load thread are captured into the payload as
  `error` instead of vanishing.
- **Correlation is the deliverable.** The prompt tells the model how to read the
  combination: high CPU with high latency implies a CPU-bound handler, while
  high latency with low CPU implies sleeping, blocking I/O, or a downstream
  dependency. `/heavy` (hashing loop) and `/slow` (`time.sleep`) exist precisely
  so both cases can be demonstrated with the same latency.
- **`/heavy` hashes 256 KB blocks rather than digests.** A 32-byte hash chain
  holds the GIL: 8 concurrent requests measured 0.99 cores busy, so the profiler
  read ~12% CPU on an 8-core box while latency doubled from GIL contention. Above
  CPython's ~2 KB GIL-release threshold OpenSSL hashes without the GIL — the same
  8 threads use 5.3 cores in 1.5 s, and 10 concurrent requests peak at 100% CPU.
  The endpoint exists to produce a CPU signature, so it has to consume cores.



## Configuration

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `LLM_API_KEY` | Yes | — | Groq API key |
| `LLM_BASE_URL` | No | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | No | `openai/gpt-oss-120b` | Model id sent to the provider |

`python-dotenv` reads `.env` at import; real environment variables still win
because `load_dotenv()` does not override existing values. `.env` is gitignored;
the API key never enters source control or tool output.

## Design decisions

- **Direct SDK over an agent framework.** One tool and one loop do not justify
  LangChain or similar. The explicit loop is auditable and easy to extend.
- **One combined diagnostic tool, not two agent calls.** CPU and latency are only
  meaningful when measured over the same requests, so the profiler and the load
  test run in one call and return one payload. A model that had to call two tools
  could pair a CPU reading with an unrelated test.
- **Threading rather than multiprocessing.** The load test is I/O-bound
  (`requests` + a 10-worker thread pool), so a background `threading.Thread`
  overlaps it with the profiler without pickling costs. The profiler's blocking
  `cpu_percent(interval=1)` call is what makes the main thread a natural sampler.
- **Local tool execution.** The load generator shares a process with the agent,
  which is fine for Phase 3 but means agent and load-driver CPU are inside the
  machine-wide CPU figure.
- **Fixed load shape.** Concurrency (10 workers) and the interactive timeout (3 s)
  are constants in `tools.py`; `run_full_diagnostic` raises the per-request
  deadline to 15 s so CPU-bound latency is not misfiled as a client failure. The
  profile window is a keyword argument (`profile_seconds`, default 5) that the
  orchestrator currently does not expose to the model.
- **Failures as data.** HTTP-level failures are reported, not raised. Only
  configuration errors (missing key), invalid tool arguments, and a runaway tool
  loop abort a run.

## Known limitations

- No persistence: each run's metrics are printed and discarded.
- No rate limiting, ramp-up, or workload scenarios — a single burst only.
- Synchronous request path; the orchestrator blocks until the model answers.
- CPU/memory are machine-wide and sampled on a fixed window, not per-process and
  not per-request; a load test longer than the window is only partially covered.
- The target API is a local FastAPI app with no database, so database bottleneck
  detection still depends on future target-side instrumentation.

