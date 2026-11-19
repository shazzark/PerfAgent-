# PerfAgent

**An AI agent that load-tests your API, reads the numbers, and tells you where
the bottleneck is.**

PerfAgent gives a large language model a real performance-testing tool instead
of asking it to guess. You describe a target in plain English; the agent runs a
concurrent load generator while sampling CPU and memory, then answers with an
evidence-based read on performance and a concrete fix for the bottleneck it
found.

````text
You:  Diagnose http://127.0.0.1:8000/heavy with 12 requests.
Agent: [calls run_full_diagnostic(url=..., total_requests=12)]
       -> {"load_test": {"total_time_seconds": 3.12, "successful": 12,
                         "avg_latency_seconds": 1.591, "max_latency_seconds": 1.613},
           "baseline": {"avg_cpu_percent": 37.9},
           "system_metrics": {"avg_cpu_percent": 61.6, "peak_cpu_percent": 96.4},
           "cpu_delta_percent": 23.7, "memory_delta_percent": -0.31}
Agent: Each request took ~1.6 s and all 12 succeeded. CPU went from a 37.9% idle
       baseline to a 96.4% peak while memory stayed flat (delta -0.3 pp), so the
       handler is CPU-bound: eight threads for twelve requests means the requests
       queue for cores, which is why latency is ~1.6 s per request instead of
       milliseconds. The same burst against /slow kept CPU at 27% peak, i.e. the
       time there is spent sleeping rather than computing.

       ## Suggested fix

       Reduce Gunicorn workers to match the number of logical CPUs, eliminating
       CPU oversubscription.

       ```diff
       - workers = 10
       + workers = 8
       ```
````

## Goal

Manual performance testing is slow and easy to misread: you run an ad-hoc script,
stare at a wall of numbers, and still have to decide whether 6 req/s is a client
limit or a server bottleneck. PerfAgent's goal is to close that loop —
**run a realistic load test, correlate the metrics with application behaviour,
and produce a concrete, ranked explanation of what is slow and what to change**,
then verify the fix by re-running the same test.

## How it works

PerfAgent is a tool-calling loop built on the OpenAI Python SDK:

1. The orchestrator sends your prompt, a system prompt, and the
   `run_full_diagnostic` tool schema to the model.
2. If the model requests the tool, the orchestrator executes it locally and
   returns the JSON result as a `tool` message.
3. The model analyses the metrics and either asks for another tool call or
   answers. The loop is capped at 5 rounds to prevent runaway calls. That answer
   is the diagnosis.
4. If a diagnostic actually ran, the orchestrator resolves the file that declares
   the measured route (explicit `source_file` -> `PERFAGENT_SOURCE_FILE` -> a
   project-root scan for the exact route decorator), reads it, and passes the
   diagnosis plus that source to `suggester.suggest_code_fix()`. It returns
   **Diagnosis** followed by **Suggested fix** — one narrow model call whose only
   job is a directly applicable diff, command, or statement written against the
   framework the code is actually built on.

The remediation is *not* exposed to the model as a selectable tool: the
orchestrator calls it deterministically once the diagnosis exists, so the fix is
produced exactly once, always from the model's own final analysis, and never
from a half-formed intermediate thought. Conversational answers and failed
diagnostics skip that stage entirely, and a provider error there is reported
inline instead of discarding measurements that already succeeded.

The remediation call also receives the target's source code, because a diagnosis
says *why* a handler is slow but nothing about *what it is written in*. Before
that context was injected, a live run guessed Flask and diffed a `Blueprint` in
a non-existent `app/routes.py` for a service that is a 95-line FastAPI module.
`source_code_context` is therefore a required argument of
`suggest_code_fix()`, and `agent.py` resolves it from the URL the tool actually
measured. If the file cannot be resolved or read, the diagnosis is still
returned with a framework-agnostic fix and an explicit note that the source was
unavailable - it never silently reverts to a guess.

The tool exposed today is `run_full_diagnostic(url, total_requests)`. It starts
`run_load_test` (10 workers, 15 s diagnostic deadline) on a background thread and
samples machine CPU and memory with `psutil` over a 5 s window in the foreground,
then returns per-request latency (average, p95, max), success/failure counts,
requests per second, and average/peak CPU and memory percentages for the same run,
alongside a 2 s idle baseline and the load-minus-baseline deltas. `run_load_test`
remains available for latency-only runs with its 3 s interactive deadline.


## Status

Phases 1–5 are complete. Phase 4 — **AI Analysis & Suggestion** — was verified
end to end: a 12-request burst on `/heavy` raised CPU from a 31.7% idle baseline
to a 91.1% peak (+27.8 points) while memory stayed flat, and the model named the
handler CPU-bound. Phase 4.5 injects the target's source code, which resolved
`/heavy` to `dummy_api.py` and stopped the framework hallucination (a Flask
`Blueprint` for a FastAPI service) seen before it.

Phase 5 — **Feedback Loop** — was verified live too, with a result worth reading:
the model's `ProcessPoolExecutor` rewrite was applied to a staging copy, re-tested
with the same 12-request burst, and **regressed** it — average latency 1532 ms →
3322 ms, p95 1572 ms → 4787 ms, throughput 3.94 → 2.44 req/s, CPU delta +9.3
points, no failures. The agent said so, deleted the copy, and restarted the
original server; `dummy_api.py` never changed. A confident-sounding fix that had
never been measured is exactly what this phase exists to catch. Target-side
tracing, query timing, and metric persistence are still open (see
[ROADMAP.md](ROADMAP.md)).


## Requirements

- Python 3.13
- A Groq API key (free tier works) for an OpenAI-compatible endpoint
- Windows, macOS, or Linux

## Setup

```powershell
# 1. Install dependencies
py -m pip install openai python-dotenv fastapi uvicorn requests psutil

# 2. Create .env in the project root (never commit it)
@"
LLM_API_KEY=your_groq_api_key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
"@ | Set-Content -LiteralPath .env
```

`LLM_BASE_URL` and `LLM_MODEL` are optional; they default to Groq and
`openai/gpt-oss-120b`.

## Run it

Open two terminals in the project root.

**Terminal 1 — start the target API:**

```powershell
py -m uvicorn dummy_api:app --reload
```

`/health` responds instantly, `/slow` sleeps 1.5 s to simulate a blocked handler,
and `/heavy` burns about 1.5 s of real CPU hashing 256 KB blocks, so concurrent
requests spread across all cores. `/slow` and `/heavy` have similar latency but
very different CPU signatures — that contrast is how you can tell an agent reading
is real.

**Terminal 2 — start the agent:**

```powershell
py agent.py "Diagnose http://127.0.0.1:8000/heavy with 10 requests and tell me if it is CPU-bound"
```

Running `py agent.py` with no arguments uses the built-in demo prompt against
`/heavy`. Before asking for a fix, the agent looks for the project file that
declares the measured route (`@app.get("/heavy")` in `dummy_api.py`) and sends its
source with the diagnosis, so the patch matches the real framework. Point it at a
different file when the target lives outside the project root or behind a path
decorator the scan cannot see:

```powershell
py agent.py --source path/to/app.py "Diagnose http://127.0.0.1:8000/heavy"
$env:PERFAGENT_SOURCE_FILE = "path/to/app.py"   # same override, via the environment
```

The tools can also be exercised without the LLM:

```powershell
py profiler.py     # 5 s of idle-machine CPU/memory samples
py tools.py        # smoke-tests /health, /slow, then a full diagnostic on /heavy
```

### Verifying a suggested fix

When a fix is suggested, an interactive run asks
`Would you like to apply this fix to staging and verify it? (y/n)`. Answering `y`
copies `dummy_api.py` to `staging_api.py`, asks the model to rewrite that copy
with the fix applied, restarts uvicorn against the copy on the same port,
re-runs the diagnostic, prints a before/after table, deletes the copy and
restarts your original server.

```powershell
py agent.py --verify "Diagnose http://127.0.0.1:8000/heavy with 12 requests"
```

The measured run's output (a fix that made things worse, so nothing was applied):

```text
🧪 Verifying the fix on a staging copy of dummy_api.py...
📝 Staged staging_api.py (4008 chars); dummy_api.py was not modified.
🛑 Stopping the server on port 8000...
🚀 Starting staging_api:app on port 8000...
🧹 Removed staging_api.py.
♻️  Restarting dummy_api:app on port 8000...
🧾 Verdict: regressed - average latency rose 116.8%

| Metric | Before | After | Change (after vs before) |
| --- | --- | --- | --- |
| Avg latency | 1532 ms | 3322 ms | +116.8% |
| p95 latency | 1572 ms | 4787 ms | +204.5% |
| Throughput | 3.94 req/s | 2.44 req/s | -38.1% |
| Failed requests | 0 | 0 | +0 |
| CPU delta | 25.65 pts | 34.95 pts | +9.3 pts |

**Verdict: regressed** - average latency rose 116.8%.
```

`--verify` skips the prompt (for scripts and unattended runs); `--no-verify` will
never ask. The prompt is also skipped when stdin is not a terminal, so a piped or
CI run cannot block on `input()`. Lower latency and higher throughput are better,
so a negative latency change is a win; only changes of 5% or more are called real,
and a new failed request always counts as a regression. The measured file is never
written — if the copy does not parse, does not assign `app`, or drops the measured
route, the run is aborted before any server starts.

The remediation stage has its own smoke test, which spends exactly one model call
on a canned CPU-bound diagnosis against a small FastAPI source snippet:

```powershell
py suggester.py    # -> one short lead-in plus a fenced diff/command
```

The orchestrator contract (diagnosis -> suggestion, the cases where no suggestion
runs, and how the target's source code is resolved and attached) is covered by
tests that script the model, so they need no API key, no target server, and no
network:

```powershell
py -m unittest discover -s tests -t . -v
```


## Project structure

```text
PerfAgent/
├── agent.py            Orchestrator: tool loop, source lookup, suggested fix, --verify
├── suggester.py        suggest_code_fix() + suggest_staged_file() - one model call each
├── feedback_loop.py    Staging copy, restart cycle, before/after comparison
├── llm_client.py       Shared model id, base URL, and client factory
├── tools.py            run_load_test() and run_full_diagnostic()
├── profiler.py         get_system_metrics() - CPU/memory sampling via psutil
├── load_test_tool.py   Deprecated re-export kept for Phase 1 compatibility
├── dummy_api.py        FastAPI target: /health, /slow (sleep), /heavy (CPU-bound)
├── tests/              Unittest suite: orchestration, suggester, staging, comparison
├── list_models.py      Diagnostics: list models available to your key
├── test_connection.py  Diagnostics: one chat completion smoke test
├── ROADMAP.md          Five-phase delivery plan
├── ARCHITECTURE.md     Stack, components, tool contract, design decisions
├── README.md           This file
└── .gitignore          Excludes .env, caches, generated JSON, and the staging copy
```

## What the numbers mean

| Field | Meaning |
| --- | --- |
| `load_test.total_time_seconds` | Wall-clock time for the whole burst |
| `load_test.avg_latency_seconds` / `p95_latency_seconds` / `max_latency_seconds` | Per-request latency; the only correct latency figure, since requests run 10-wide |
| `load_test.successful` / `failed` | HTTP 200 vs. timeout, connection error, or other status |
| `load_test.requests_per_second` | `total_requests / total_time_seconds` |
| `baseline.*` | Idle CPU/memory sampled for 2 s immediately before the test |
| `system_metrics.avg_cpu_percent` | Mean CPU across all logical CPUs in the window (100 = all cores busy) |
| `system_metrics.avg_memory_percent` | Mean system RAM in use during the window |
| `system_metrics.peak_cpu_percent` / `peak_memory_percent` | Highest single-second reading |
| `system_metrics.cpu_samples` / `memory_samples` | The per-second series behind the averages |
| `cpu_delta_percent` / `memory_delta_percent` | Load average minus baseline average |
| `profile_covers_full_load_test` | `false` when the load test outlasted the sampling window |

Read them together: high latency with high CPU points at a CPU-bound handler;
high latency with low CPU points at sleeping, blocking I/O, or a downstream
dependency. CPU is machine-wide, so the load generator's own work is included —
compare the load window against the baseline instead of reading the absolute
number. Never divide `total_time_seconds` by `total_requests`: the burst runs ten
requests at a time, so that quotient is roughly the per-request latency divided
by ten.

Failures are reported rather than raised: under load, a timeout is a result.
Missing configuration (no `LLM_API_KEY`) and a model that loops past 5 tool
rounds are the only conditions that abort a run.

## Roadmap at a glance

1. **The Muscle (Tools) — completed.** Dummy API, concurrent load generator.
2. **The Brain (Orchestrator) — completed.** LLM tool calling via Groq.
3. **The Eyes (Profiler) — completed.** CPU and memory correlated with load.
4. **AI Analysis — completed.** Root-cause ranking plus one concrete fix.
5. **Feedback Loop — completed.** Apply the fix to a staging copy, re-test, compare.

Full detail and checklists live in [ROADMAP.md](ROADMAP.md); technical decisions
are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Limitations

- Single-burst load only — no ramp-up, duration, or mixed scenarios.
- No result persistence or comparison between runs yet.
- Suggested fixes are model output, and only the file that declares the measured
  route is ever staged. Verification covers one endpoint, one burst size, and one
  run: a fix that helps `/heavy` may hurt `/slow`, and a ±few percent reading is
  inside host noise. A staged rewrite that the model truncates or mangles is
  rejected by validation instead of being measured, so a rejected fix is not
  evidence that it was wrong. The staging server replaces the one on the measured
  port for the duration of the re-test, so nothing else can use that port, and the
  original is restarted without `--reload`.
- Nothing is deployed anywhere: the loop writes, serves, measures and deletes a
  local copy. Applying a verified fix to the real file is still a manual step.
- CPU and memory are machine-wide and sampled on a fixed 5 s window, not
  per-process or per-request.
- The bundled target API has no database, so database bottleneck detection
  depends on future target-side instrumentation.
- The load generator runs in the same process as the agent; results are not
  clean benchmarks on a loaded machine.


## Security notes

`.env` is gitignored and the API key is read only through `python-dotenv`. Do not
paste keys into `agent.py`, issues, or documentation. Only load-test systems you
own or are authorised to test.

Staging writes one file (`staging_api.py`, in the project root) and deletes it in a
`finally`. It will not overwrite a `staging_api.py` that already exists, and it
refuses to stop a process on the target port unless that process is a uvicorn
server this run started — anything else raises instead of being killed. Model
output is parsed and compiled before it is served, and it is never written over the
file that was measured.
