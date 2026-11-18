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
| Timing | `time.perf_counter()` | Monotonic wall-clock measurement |

## Components

```
agent.py            Orchestrator: prompt -> LLM -> tool call -> LLM -> answer
tools.py            run_load_test(url, total_requests) -> metrics dict
load_test_tool.py   Deprecated re-export of run_load_test (Phase 1 compatibility)
dummy_api.py        FastAPI target: /health (instant) and /slow (1.5 s sleep)
list_models.py      Diagnostics: prints models available to the Groq key
test_connection.py  Diagnostics: single chat completion against the configured model
```

Each file has one job. `agent.py` owns conversation state and LLM I/O;
`tools.py` owns measurement and has no knowledge of the LLM.

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
   │                                  tools.run_load_test(...)
   │                                             │
   │                                  metrics dict (JSON)
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
  "name": "run_load_test",
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
  "url": "http://127.0.0.1:8000/slow",
  "total_requests": 20,
  "total_time_seconds": 3.12,
  "successful": 20,
  "failed": 0,
  "requests_per_second": 6.41
}
```

A success is an HTTP `200` response within 3 seconds. Timeouts, connection
errors, and other status codes count as failures and do **not** raise, because
under load those are measurements, not exceptions. Invalid arguments are
returned as `{"error": "..."}` so the model can self-correct.

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
- **Local tool execution.** The load generator shares a process with the agent,
  which is fine for Phase 2 but means agent CPU competes with the load driver.
- **Fixed load shape.** Concurrency (10 workers) and timeout (3 s) are constants
  in `tools.py`; Phase 3 will make them inputs once the profiler needs varied
  profiles.
- **Failures as data.** HTTP-level failures are reported, not raised. Only
  configuration errors (missing key) and a runaway tool loop abort a run.

## Known limitations

- No persistence: each run's metrics are printed and discarded.
- No rate limiting, ramp-up, or workload scenarios — a single burst only.
- Synchronous request path; the orchestrator blocks until the model answers.
- The target API is a local FastAPI app with no database, so Phase 3 profiler
  work needs a more realistic service to instrument.
