# PerfAgent

**An AI agent that load-tests your API, reads the numbers, and tells you where
the bottleneck is.**

PerfAgent gives a large language model a real performance-testing tool instead
of asking it to guess. You describe a target in plain English; the agent calls a
concurrent load generator, receives measured throughput and failure counts, and
answers with an evidence-based read on performance.

```text
You:  Load test http://127.0.0.1:8000/slow with 20 requests.
Agent: [calls run_load_test(url=..., total_requests=20)]
       -> {"total_time_seconds": 3.12, "successful": 20, "failed": 0, "rps": 6.41}
Agent: Throughput is ~6.4 req/s against a 10-worker pool, so each request is
       waiting on the server rather than the client. 1.5 s of per-request
       latency looks like a blocking call in the handler...
```

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
   `run_load_test` tool schema to the model.
2. If the model requests the tool, the orchestrator executes it locally and
   returns the JSON result as a `tool` message.
3. The model analyses the metrics and either asks for another tool call or
   answers. The loop is capped at 5 rounds to prevent runaway calls.

The only capability exposed today is `run_load_test(url, total_requests)`,
which fires a burst of concurrent `GET` requests (10 workers, 3 s timeout) and
reports total time, successes, failures, and requests per second.

## Status

Phase 2 — **The Brain** is in progress: the tool schema, provider connection,
tool execution, and result hand-back are implemented and verified. Profiling and
autonomous remediation are future phases (see [ROADMAP.md](ROADMAP.md)).

## Requirements

- Python 3.13
- A Groq API key (free tier works) for an OpenAI-compatible endpoint
- Windows, macOS, or Linux

## Setup

```powershell
# 1. Install dependencies
py -m pip install openai python-dotenv fastapi uvicorn requests

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

`/health` responds instantly; `/slow` sleeps 1.5 s to simulate a bottleneck.

**Terminal 2 — start the agent:**

```powershell
py agent.py "Load test http://127.0.0.1:8000/slow with 20 requests and tell me what is slow"
```

Running `py agent.py` with no arguments uses the built-in demo prompt. The tool
can also be exercised without the LLM:

```powershell
py tools.py        # smoke-tests /health and /slow
```

## Project structure

```text
PerfAgent/
├── agent.py            Orchestrator: LLM tool-calling loop (Phase 2)
├── tools.py            run_load_test() - the agent's only tool
├── load_test_tool.py   Deprecated re-export kept for Phase 1 compatibility
├── dummy_api.py        FastAPI target with /health and /slow endpoints
├── list_models.py      Diagnostics: list models available to your key
├── test_connection.py  Diagnostics: one chat completion smoke test
├── ROADMAP.md          Five-phase delivery plan
├── ARCHITECTURE.md     Stack, components, tool contract, design decisions
├── README.md           This file
└── .gitignore          Excludes .env, caches, and generated JSON
```

## What the numbers mean

| Field | Meaning |
| --- | --- |
| `total_time_seconds` | Wall-clock time for the whole burst |
| `successful` / `failed` | HTTP 200 vs. timeout, connection error, or other status |
| `requests_per_second` | `total_requests / total_time_seconds` |

Failures are reported rather than raised: under load, a timeout is a result.
Missing configuration (no `LLM_API_KEY`) and a model that loops past 5 tool
rounds are the only conditions that abort a run.

## Roadmap at a glance

1. **The Muscle (Tools) — completed.** Dummy API, concurrent load generator.
2. **The Brain (Orchestrator) — in progress.** LLM tool calling via Groq.
3. **The Eyes (Profiler).** CPU, memory, traces, and query timing per run.
4. **AI Analysis.** Root-cause ranking and remediation suggestions.
5. **Feedback Loop.** Apply fixes in staging, re-test, compare.

Full detail and checklists live in [ROADMAP.md](ROADMAP.md); technical decisions
are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Limitations

- Single-burst load only — no ramp-up, duration, or mixed scenarios.
- No result persistence or comparison between runs yet.
- The bundled target API has no database, so database bottleneck detection
  depends on Phase 3 instrumentation.
- The load generator runs in the same process as the agent; results are not
  clean benchmarks on a loaded machine.

## Security notes

`.env` is gitignored and the API key is read only through `python-dotenv`. Do not
paste keys into `agent.py`, issues, or documentation. Only load-test systems you
own or are authorised to test.
