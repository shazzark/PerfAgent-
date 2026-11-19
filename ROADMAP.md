# PerfAgent Roadmap

## Phase 1: The Muscle (Tools - Completed)

- [x] Dummy target API with a healthy and a deliberately slow endpoint (`dummy_api.py`)
- [x] Concurrent load-testing tool (`tools.py`, re-exported by `load_test_tool.py`)
- [x] Throughput, latency, and failure reporting verified against the dummy API

## Phase 2: The Brain (Orchestrator - Completed)

- [x] Connect to Groq through the OpenAI Python SDK
- [x] Expose the measurement tool to the model as a tool schema
- [x] Execute tool calls and feed results back for analysis
- [x] Harden the prompt and error handling based on live runs

## Phase 3: The Eyes (Profiler - Completed)

- [x] Add a CPU-bound `/heavy` endpoint to the dummy target API (time-boxed 256 KB hashing loop, ~1.5 s per request, spread across cores)
- [x] Machine-level CPU and memory profiler (`profiler.py`, `psutil`)
- [x] `run_full_diagnostic` runs the load test on a background thread while the profiler samples the same window, returning one combined payload
- [x] Update the agent tool schema and system prompt so latency is correlated with CPU and memory

Deferred past Phase 4, tracked so they are not lost:

- [ ] Instrument the target API to capture per-request CPU and request traces
- [ ] Correlate database query time with endpoint response latency
- [ ] Persist raw metrics for every load-test run

Current Phase 3 limitation: the profiler measures the whole machine, not the
target process, so a load generator sharing the host contributes to the CPU
figure. On a host that is already busy the idle baseline is high, the load
window has no headroom left, and the CPU delta can no longer separate the
target from unrelated software. The fixed 5 s sampling window can also be
shorter than a long load test; `profile_covers_full_load_test` reports when that
happened.


## Phase 4: AI Analysis & Suggestion (Completed)

- [x] Feed profiling data to the LLM for root-cause analysis
- [x] Rank bottlenecks by measured impact
- [x] Produce concrete remediation suggestions (indexes, caching, pooling)
- [x] Deterministic second call (`suggester.suggest_code_fix`) so every measured diagnosis ends in one fenced diff, command, or SQL statement
- [x] Gate the suggestion on a diagnostic that actually produced load numbers, so conversational answers pass through unchanged
- [x] Report a suggestion-provider failure inline instead of discarding the measurements
- [x] Contract tests for the diagnosis -> suggestion wiring (`tests/`, scripted fake client)
- [x] Verify live: 12 requests on `/heavy` moved CPU from a 31.7% idle baseline to a 91.1% peak (+27.8 points) with flat memory, the model returned a CPU-bound verdict, and the fix was a `ProcessPoolExecutor` diff

Deferred past Phase 4:

- [ ] Per-process CPU sampling for the target, so a busy dev machine cannot mask the CPU signal (the live run only separated cleanly because the host had headroom: baseline 31.7%)


## Phase 4.5: Repository Context for the Suggester (Completed)

The Phase 4 live run diagnosed `/heavy` correctly but proposed a Flask
`Blueprint` diff against a non-existent `app/routes.py`, because the suggester
saw the diagnosis and nothing else. A patch cannot be applied safely to a
codebase the model has never read, so the context gap was closed before Phase 5.

- [x] Make `source_code_context` a required argument of `suggester.suggest_code_fix()`, so no caller can request a fix blindfolded
- [x] Inject the target's source, fenced and verbatim, into `FIX_PROMPT_TEMPLATE`, and instruct the system prompt to match that framework and never substitute another
- [x] Resolve the file from the URL the tool actually measured: explicit `source_file` -> `PERFAGENT_SOURCE_FILE` -> project-root scan for the exact route decorator (`agent.find_source_file`)
- [x] Degrade explicitly: an unresolvable or unreadable file sends `NO_SOURCE_CONTEXT` and the fix is framework-agnostic, while the completed diagnosis is still reported
- [x] Cap the attached source (`MAX_SOURCE_CONTEXT_CHARS`) so a large module cannot crowd out the remediation prompt
- [x] Tests for the prompt contract (context fenced verbatim, blank context marked unknown, required argument) and the resolution path (route -> file, override, env var, unknown route, truncation)
- [x] Verify live: `/heavy` resolved to `dummy_api.py` (3,795 chars), and the suggested diff stayed on FastAPI - `async def heavy_endpoint`, `asyncio.get_running_loop()`, `ProcessPoolExecutor`, reusing `_burn_cpu` and `CPU_BURN_SECONDS` - instead of the Phase 4 Flask guess


## Phase 5: Feedback Loop

- [ ] Apply proposed fixes in a Docker staging environment
- [ ] Re-run the same load test and compare before/after metrics
- [ ] Report a regression-checked summary of what changed and what improved
