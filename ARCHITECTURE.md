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
| Testing | `unittest` (stdlib) + scripted fake client | Orchestration contract tests without network or SDK cost |

## Components

```
agent.py            Orchestrator: prompt -> LLM -> tool call -> LLM -> answer,
                    then either Local source lookup + suggest_code_fix() +
                    optional staging verification, or a Remote Audit report
suggester.py        suggest_code_fix(diagnosis, source) -> one fenced remediation
                    suggest_staged_file(source, fix) -> the whole fixed file
feedback_loop.py    Staging verification: copy -> fix -> serve -> re-test -> compare
llm_client.py       Shared MODEL, BASE_URL and create_client() factory
tools.py            run_load_test(url, total_requests) -> load metrics
                    run_full_diagnostic(url, total_requests) -> load + system metrics
                    find_breaking_point(url, start_rps, step_increase, max_rps)
                    -> breaking/last-clean RPS + per-step metrics
profiler.py         get_system_metrics(duration_seconds) -> avg CPU/memory
load_test_tool.py   Deprecated re-export of run_load_test (Phase 1 compatibility)
dummy_api.py        FastAPI target: /health (instant), /slow (1.5 s sleep),
                    /heavy (1.5 s of real CPU-bound hashing)
tests/              Unittest suite: orchestration, suggester, source context
list_models.py      Diagnostics: prints models available to the Groq key
test_connection.py  Diagnostics: single chat completion against the configured model
```

Each file has one job. `agent.py` owns conversation state, LLM I/O, and the
order of the two model calls; `suggester.py` owns the remediation prompt and
never measures; `feedback_loop.py` owns the staging process lifecycle and never
writes the measured file; `llm_client.py` owns configuration so both callers
share one client without a circular import; `tools.py` owns measurement
orchestration; `profiler.py` owns resource sampling; no tool or profiler module
knows the LLM exists.


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

   no tool_calls and diagnostic_ran == False  ->  return the answer unchanged
   no tool_calls and diagnostic_ran == True
                                            |
                                            ▼
   _with_suggested_fix(diagnosis, client, source_context)
      (source_path + context resolved once in run_agent: explicit -> env -> route)
      └─ suggester.suggest_code_fix(diagnosis, source)  ──►  Groq chat.completions
                                            |
                                            ▼
   "**Diagnosis** <analysis>  ## Suggested fix <fenced code>"
                                            |
                          verify? prompt / --verify / --no-verify
                                            |
                                            ▼
   _maybe_verify()  ->  feedback_loop.run_feedback_loop()
      ├─ suggest_staged_file()          whole file, fix applied
      ├─ validate_staged_source()       parses + `app` + measured route
      ├─ LocalServer: stop -> start("staging_api") on the same port
      ├─ run_full_diagnostic(url, n)    identical burst
      ├─ compare_metrics(before, after) verdict + deltas
      └─ finally: delete staging_api.py -> start("dummy_api")
```


The assistant message containing `tool_calls` is appended before its tool
results. OpenAI-compatible APIs reject a `tool` message whose `tool_call_id`
does not belong to a preceding assistant turn.

`find_breaking_point` is a second entry in the same loop, not a separate agent:
`_execute_tool` dispatches by name, the tool's per-step JSON returns as the `tool`
message, and the model explains it. Each step runs a paced one-second burst
through the same 10-worker pool (`_run_ramp_step`), summarises HTTP latency,
failures, connection failures, and worker queue wait (`_summarize_ramp_burst`),
and the loop stops only at the first step that fails 100% of its requests or
reports a connection failure. Latency and partial failure never end the ramp;
they are the evidence collected on the way to the crash. `step_increase=None`
doubles the offered rate after each surviving step (10 -> 20 -> 40 -> 80), and an
explicit integer switches to additive steps for a finer table.
`ramp_ran` short-circuits the final answer before both the Audit report and the
Local suggester: a capacity report has no `load_test` payload for the audit
builder and no bottleneck that justifies a code fix.

## Modes: Local vs Remote Audit

The measured URL decides what happens after the tool loop. There are two modes:

```
first URL in prompt ──► loopback? ──► yes ──► Local mode
        │                    │
        │ no                 └──► no URL: wait for the measured URL
        ▼
   Remote Audit mode
```

- **Local** (`localhost`, `*.localhost`, or any loopback IP): the run resolves the
  file that declares the measured route, calls `suggest_code_fix()`, and may stage
  and verify the result. This is the only mode that can propose code.
- **Remote Audit** (every other host, including private LAN addresses): the same
  `run_full_diagnostic` tool measures the API, but `source_file_for`,
  `load_source_context`, `suggest_code_fix`, and `run_feedback_loop` are not
  reachable. The final answer is `build_audit_report(analysis, result)`.
- If the prompt has no URL, `detect_mode` defaults to Local so the tool schema and
  wording are usable, then `run_agent` switches to Audit mode when the measured
  URL is remote and appends `AUDIT_MODE_NOTICE` to the next request. The branch
  that builds the report is guarded by `audit_mode`, not by the notice, so a model
  that ignores the notice still cannot reach the suggester.
- `mode=MODE_LOCAL|MODE_AUDIT` forces a mode for callers and tests; any other
  value raises `ValueError`. The CLI does not expose a flag because the URL is
  already the input that determines the mode.
- Ramp-up is mode-independent: both system prompts advertise
  `find_breaking_point`, and a completed ramp returns the model's capacity
  analysis directly. It never reaches the remote audit builder (which would look
  for a missing `load_test` and report `Inconclusive`) or the local suggester
  (a breaking point is a capacity limit, not necessarily a code defect).

The Audit verdict is computed, not generated:

| Check (in order) | Verdict | Reason text |
| --- | --- | --- |
| `load_test` missing/empty | `Inconclusive` | diagnostic produced no measurements |
| `failed > 0` | `Broken` | failed / total and failure rate |
| `p95_latency_seconds > 2.0` | `Degraded` | p95 above threshold, no failures |
| otherwise | `Stable` | no failures, p95 within threshold |

`build_audit_report` also renders target, failure rate, latency, throughput, and
the load-generator CPU delta from the payload, then appends the model's cleaned
prose as `### Analysis`. `_clean_audit_analysis` removes any model-authored
`**Verdict:**` line and truncates at a `Suggested Fix` heading, so prompt
compliance is not part of the correctness argument.

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

Ramp schema (only `url` is required; omitted rates use the function defaults
`start_rps=10`, `step_increase=None` (double each step), `max_rps=1000`):

```json
{
  "name": "find_breaking_point",
  "parameters": {
    "type": "object",
    "properties": {
      "url": { "type": "string" },
      "start_rps": { "type": "integer", "minimum": 1 },
      "step_increase": { "type": "integer", "minimum": 1 },
      "max_rps": { "type": "integer", "minimum": 1 }
    },
    "required": ["url"]
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

Ramp payload (a live `/slow` run that never died):

```json
{
  "url": "http://127.0.0.1:8000/slow",
  "start_rps": 5,
  "step_increase": null,
  "max_rps": 80,
  "strategy": "double",
  "found": false,
  "breaking_rps": null,
  "last_successful_rps": 80,
  "crash_reason": null,
  "breaking_reason": null,
  "final_metrics": {
    "offered_rps": 80,
    "total_requests": 80,
    "successful": 80,
    "failed": 0,
    "failure_rate_percent": 0.0,
    "failure_breakdown": {},
    "connection_failures": 0,
    "all_failed": false,
    "avg_latency_seconds": 1.511,
    "p95_latency_seconds": 1.524,
    "max_latency_seconds": 1.531,
    "end_to_end_avg_latency_seconds": 6.365,
    "client_queue_avg_seconds": 4.854,
    "client_queue_max_seconds": 9.707,
    "client_limited": true
  },
  "breaking_point": null,
  "steps": ["every attempted step, same shape as final_metrics"],
  "stop_conditions": {
    "stop_on_total_failure": true,
    "stop_on_connection_error": true,
    "failure_rate_percent": 100.0
  },
  "client_concurrency_ceiling": 10,
  "break_was_client_limited": true
}
```

And a live crash payload (closed port, first step):

```json
{
  "url": "http://127.0.0.1:8999/slow",
  "start_rps": 5,
  "step_increase": null,
  "max_rps": 1000,
  "strategy": "double",
  "found": true,
  "breaking_rps": 5,
  "last_successful_rps": null,
  "crash_reason": "Connection Refused",
  "breaking_reason": "Connection Refused",
  "final_metrics": {
    "offered_rps": 5,
    "total_requests": 5,
    "successful": 0,
    "failed": 5,
    "failure_rate_percent": 100.0,
    "failure_breakdown": { "connection_refused": 5 },
    "connection_failures": 5,
    "all_failed": true,
    "avg_latency_seconds": 2.028,
    "p95_latency_seconds": 2.054,
    "max_latency_seconds": 2.054,
    "end_to_end_avg_latency_seconds": 2.028,
    "client_queue_avg_seconds": 0.001,
    "client_queue_max_seconds": 0.002,
    "client_limited": false
  },
  "breaking_point": "(alias of final_metrics)",
  "steps": ["one step"],
  "client_concurrency_ceiling": 10,
  "break_was_client_limited": false
}
```

`avg_latency_seconds` is HTTP time only. `end_to_end_avg_latency_seconds` adds
`client_queue_avg_seconds`, the wait before one of the 10 workers starts the
request. Keeping the two separate is what lets the report say "the API was still
at 1.51 s, but the client queue pushed the observed wait to 6.37 s" instead of
attributing generator queueing to the server. Latency alone never ends the ramp;
when no step fails completely, the payload has `found: false`,
`breaking_rps: null`, `crash_reason: null`, `breaking_point: null`, and
`last_successful_rps` set to `max_rps`. `failure_breakdown` names each failure
mode (`HTTP 500`, `timeout`, `connection_refused`, `connection_error`), and
`connection_failures`/`all_failed` are derived from it so the loop never has to
guess why a step died.

A success is an HTTP `200` response inside the client deadline. The interactive
default is 3 seconds; `run_full_diagnostic` passes
`DIAGNOSTIC_TIMEOUT_SECONDS = 15` because a CPU-bound handler legitimately takes
longer per request once concurrency oversubscribes the cores, and counting that
latency as a client-side failure would hide the signal being profiled. Timeouts,
connection errors, and other status codes still count as failures and do **not**
raise, because under load those are measurements, not exceptions. Invalid
arguments are returned as `{"error": "..."}` so the model can self-correct.


`run_load_test(url, total_requests)` remains available for latency-only runs
and for the deprecated `load_test_tool.py` compatibility module; the agent
advertises `run_full_diagnostic` and `find_breaking_point`.

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



## Suggester (`suggester.py`)

`suggest_code_fix(diagnosis_summary, source_code_context, client=None)` makes
exactly one completion call: a system prompt that fixes the output shape, and a
user message carrying both the diagnosis and the target's source code verbatim
inside `FIX_PROMPT_TEMPLATE`. It returns the model's reply — at most two
sentences of lead-in plus one fenced code block — after rejecting an empty
diagnosis (`ValueError`), a non-string context (`ValueError`), or an empty reply
(`RuntimeError`).

`source_code_context` has no default and is deliberately a required positional
argument. Phase 4 sent the diagnosis alone, and the first live run invented a
Flask `Blueprint` patch for a FastAPI module — a fix that could not be applied.
A default would let that bug return silently, so the only way to omit the code is
to pass an empty string explicitly, which the prompt reports as
`NO_SOURCE_CONTEXT` ("framework and handler signatures are unknown") instead of
letting the model guess.

### Source context resolution (`agent.py`)

`load_source_context(url, source_file)` runs before the remediation call and
returns the text handed to the suggester:

- **Precedence:** explicit `run_agent(source_file=...)` / `--source`, then
  `PERFAGENT_SOURCE_FILE`, then `find_source_file(url)`.
- **Route scan:** `find_source_file` reads only `*.py` files in the project root
  (skipping the agent's own modules) and accepts only a decorator whose path
  matches exactly, `@<obj>.<get|post|put|patch|delete|head|options|route|api_route>("/heavy")`.
  A docstring that mentions `/heavy` therefore does not count. This is
  intentionally a local, shallow search: a general AST index or router registry
  would encode assumptions about layouts this project does not own, and an
  unresolved endpoint is a valid, reported outcome.
- **Measured URL, not the prompt:** the URL comes from the successful
  `run_full_diagnostic` result (`diagnostic_url`), so the code attached is the
  file that was actually exercised even if the user's prompt named several.
- **Size cap:** `MAX_SOURCE_CONTEXT_CHARS` (16,000) truncates with an explicit
  marker, keeping the code from crowding out the remediation instruction.
- **Failure is contained:** a missing or unreadable file prints the reason,
  returns `""`, and the suggester answers without code context. Losing a
  completed measurement would be the worse failure.

The context is prefixed with `# File: <name>` so a suggested diff names the real
file instead of inventing a path.

Why the remediation is a plain function and not a second tool:

- **Deterministic, not discretionary.** A tool the model may or may not call
  makes the headline feature unreliable; a post-loop call guarantees that a
  measured diagnosis always ends in a fix.
- **The input is the finished diagnosis.** The orchestrator hands over the
  model's *final* analysis, not a mid-loop observation, and the suggestion call
  sees no tool schemas and no metric JSON to re-interpret.
- **The tool loop stays small.** One measurable tool, one safety valve. The
  remediation prompt has no tool access, so it cannot start a second loop.
- **Failures are contained.** `_with_suggested_fix` catches provider errors and
  returns the diagnosis with `_Suggestion unavailable: ..._`, because losing a
  completed measurement to a failed second call would be the worse outcome.

`SUGGEST_SYSTEM_PROMPT` also forbids repeating the diagnosis or inventing
metrics: the fix must stand on the measurement, not restate it.

## Feedback loop (`feedback_loop.py`)

Phase 4 ends with a hypothesis. Phase 5 turns it into evidence: the fix is applied
to a throwaway copy of the measured file, served on the same port, and re-measured
with the identical diagnostic. Nothing is deployed; production is out of scope by
design.

```
suggested fix + original source (read, never written)
   |
   v
suggester.suggest_staged_file()      the whole file, fix applied, fenced reply
   |
   v
validate_staged_source()             parses + `app` + measured route survives
   |  (rejects: diff body, syntax error, missing app, dropped route)
   v
staging_api.py                       written, then deleted in a `finally`
   |
   v
LocalServer.stop() -> LocalServer.start("staging_api")     same port
   |
   v
tools.run_full_diagnostic(url, n)    same burst size as the original run
   |
   v
compare_metrics(before, after)       verdict + deltas
   |
   v
stop() -> unlink(staging_api.py) -> start("dummy_api")
```

### Why the whole file instead of a diff

LLM diffs are brittle: hunk headers with invented line counts, context lines that
have drifted, three different `---`/`+++` conventions. A patch that half-applies
leaves behind a file nobody can reason about, and `patch` reports success per
hunk, so the result would have to be verified anyway. Asking for the complete file
takes the parser out of the trust boundary: the only things that must be true are
that the staged file is a valid module which still declares the measured route,
and `ast` answers both deterministically.

### Safety guardrails

Nothing here can reach production, and nothing here can silently break the
workstation:

1. **The measured file is never written.** Only `staging_api.py` is written, and
   only after validation passes.
2. **Validate before executing.** `validate_staged_source` requires a parseable
   module, a top-level `app` assignment, and the measured route path. A diff body,
   a syntax error or a fragment is rejected before any process starts.
3. **No clobbering.** An existing `staging_api.py` aborts the run instead of being
   overwritten or deleted, because it is not provably this run's artifact.
4. **Restricted kills.** `LocalServer.stop()` only terminates a process it started,
   or a listener on the target port whose command line contains `uvicorn`. Anything
   else holding that port raises instead of being killed, since a dev server on the
   wrong port is far more likely than a hostile one.
5. **Cleanup in `finally`.** The copy is unlinked and the original module restarted
   even when the re-test raises, so a failed verification cannot leave the
   workstation without a server.
6. **Confirmation needs a terminal.** The y/n prompt is skipped, with a hint to use
   `--verify`, when stdin is not a TTY - a scripted run must never hang on
   `input()`. `--no-verify` opts out explicitly.
7. **A failed verification is reported, not raised.** The diagnosis and suggested
   fix survive; the appended section says why staging could not run.

### Interpreting the comparison

`compare_metrics` calls a change real only when the average-latency improvement
clears `MATERIAL_CHANGE_PERCENT` (5%) and no new requests failed. Two identical
bursts on a shared host differ by a few percent, so a 2% "win" is reported as `no
material change`. New failures outrank any latency gain (`regressed`), and a
missing load-test payload is `inconclusive` rather than a division by zero. The
report shows avg/p95 latency, throughput, failed requests and the CPU delta in
points; the CPU figure keeps the Phase 3 machine-wide caveat, so the verdict keys
on latency and failures. The table states changes after-vs-before - a faster run
shows a negative latency change and a positive throughput change - and the verdict
supplies the interpretation, because a sign that means "better" for one metric
means "worse" for the other.


| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `LLM_API_KEY` | Yes | — | Groq API key |
| `LLM_BASE_URL` | No | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | No | `openai/gpt-oss-120b` | Model id sent to the provider |
| `PERFAGENT_SOURCE_FILE` | No | — | Overrides route-to-file detection for the remediation call |

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
- **Remediation is orchestrated, not delegated.** The fix call is sequenced by
  the orchestrator after the analysis, which costs one extra round trip but makes
  the output contract (`**Diagnosis**` then `## Suggested fix`) a property of the
  code rather than of model behaviour.
- **The suggester must read the code it patches.** Giving the model the
  diagnosis alone produced a Flask fix for a FastAPI service on the first live
  run. Sending the measured route's source is the smallest change that removes
  the guess: no repository indexing, no embeddings, no extra round trip, just the
  one file whose decorator matched. Keeping the argument required means a future
  caller cannot regress to a blind suggestion by forgetting it.
- **Degrade to honest, not to clever.** When the source cannot be located, the
  prompt says the framework is unknown and asks for a framework-agnostic fix,
  rather than omitting the context silently and inviting the model to infer a
  stack from the diagnosis wording.
- **Verification is a controlled restart, not a deployment.** The staged copy runs
  under the same uvicorn invocation the measured server used, on the same port, so
  the only variable between the two runs is the code. A container or a second port
  would add a runtime difference to a comparison that is already fighting host
  noise, and shipping to production stays a human decision.
- **The model writes a file; the orchestrator decides whether it may run.** The
  staged reply is validated structurally before it is importable, and the whole
  read/write boundary is one throwaway file. That keeps "the model suggested it"
  out of the trust boundary without pretending a parse check proves correctness.
- **One local run is not a benchmark.** The comparison refuses to call a change
  under 5% real and ranks new failures above any speedup, because the alternative -
  reporting whichever number looks best - teaches the user to trust noise.
- **A remote target is audited, never patched.** The agent cannot see a remote
  service's code, so a fix it proposed would be an unverifiable guess. Local mode
  already exists to write and re-measure code; Audit mode is the honest alternative
  for everything else. Mode selection is structural: a remote measurement disables
  the source/suggester/staging path even if the prompt or model never mentioned
  "audit".
- **The verdict is measured, the model explains it.** Asking the model for a
  verdict and trusting it would make health a function of model confidence. The
  orchestrator computes `Broken`/`Degraded`/`Stable` from `failed` and p95, strips
  a model-authored verdict, and uses the prose only for the qualitative analysis.
  The threshold lives in code so two runs of the same payload cannot disagree.
- **Failures outrank latency.** One HTTP 500 is `Broken` even when p95 is 50 ms,
  because a fast error is still an outage. Treating "most requests succeeded" as
  healthy would hide partial failures, which are the common remote symptom.
- **Ramp-up is a sequence of measured steps, not a guessed capacity number.**
  Each rate is offered for a full second and summarised before the next rate is
  tried, so `last_successful_rps` is the last rate with evidence behind it. The
  stop rules live in `tools.py`, and `crash_reason` names the exact failure that
  ended the ramp (`Connection Refused`, `100% HTTP 500s`, `100% timeouts`).
- **A client-limited ramp is labelled, not hidden.** The fixed 10-worker pool
  cannot exceed roughly 6.7 completed req/s against a 1.5 s handler, so higher
  offered rates queue without ever killing the API. The payload preserves HTTP
  latency separately and sets `client_limited`, so the report can say the target
  survived while the client was the queue. Treating queue growth as a server
  breaking point would turn a generator ceiling into a false capacity claim;
  reporting only "no break" would ignore the wait a client actually experienced.

## Known limitations

- No persistence: each run's metrics are printed and discarded.
- No configurable rate limiting or workload scenarios. Ramp-up is a fixed policy
  (one second per step, doubling by default, stop only on total failure or a
  connection error), and each step is a separate burst rather than a sustained
  test.
- Synchronous request path; the orchestrator blocks until the model answers.
- CPU/memory are machine-wide and sampled on a fixed window, not per-process and
  not per-request; a load test longer than the window is only partially covered.
- With a busy host the machine-wide baseline leaves no headroom, so the CPU delta
  cannot separate the target process from unrelated software. Process-scoped
  sampling (`psutil.Process(pid)`) is the tracked fix.
- Suggested fixes are model output. The staging loop now applies one to a copy and
  re-measures it, but it does not compile-check other modules, run the project's
  tests, or prove the rest of the app still works: one endpoint is re-tested, so a
  fix that trades this bottleneck for another endpoint's would still be reported as
  an improvement. Only the single file that declares the measured route can be
  replaced, so a fix that needs to touch helpers or configuration is applied
  partially - the copy keeps the original imports and fails at request time, which
  the comparison reports as `inconclusive` rather than as success. The remediation
  call reads just that file: routes registered through `include_router`, mounted
  sub-apps, or files outside the project root still resolve to no context (or the
  wrong file if `PERFAGENT_SOURCE_FILE` is set carelessly).
- The staging comparison is one burst on a shared host. The 5% floor bounds the
  claim, it is not a confidence interval: two runs of identical code can still land
  either side of it on a heavily loaded machine, and the CPU delta inherits the
  machine-wide caveat above.

- Audit mode measures one endpoint in one burst from the load generator's host. It
  can prove failure or slowness but cannot attribute cause: there are no server
  logs, traces, per-process CPU, or downstream timings. Its CPU/memory readings
  describe the local client, not the remote server, so they are context only. The
  2.0 s p95 threshold is a fixed policy default, not a per-target SLO.
- Ramp-up shares the fixed 10-worker client, so on a high-latency endpoint the
  generator queues and the API can survive every step; `client_limited` and
  `client_queue_avg_seconds` make that explicit. A client-limited crash rate is a
  lower bound and cannot prove the server's true capacity. The Python client is
  also capped at roughly 1,000 req/s (`DEFAULT_MAX_RPS`); 20,000+ RPS needs a
  dedicated generator such as k6 or wrk.

- The target API is a local FastAPI app with no database, so database bottleneck
  detection still depends on future target-side instrumentation.

