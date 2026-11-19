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


## Phase 5: Feedback Loop (Completed)

The original plan was a Docker staging environment. Docker was dropped: a copy of
the target served by uvicorn on the same port reproduces the feedback loop with no
image build, no mounts, and nothing left behind between runs - and it is the same
runtime the measured server already used, so the before/after comparison is not
confounded by container-vs-host behaviour. Deploying to production remains out of
scope by design; the loop only ever restarts a local process.

- [x] `feedback_loop.run_feedback_loop()` copies the measured file to `staging_api.py`, applies the fix to the copy, and never writes the original
- [x] Full-file rewrite instead of diff application (`suggester.suggest_staged_file`): LLM diffs carry invented hunk headers and drifted context, and a half-applied patch is indistinguishable from a regression
- [x] Validation gate: the staged copy must parse (`ast.parse`), assign `app`, and still declare the measured route before any server starts; a diff body or fragment is rejected
- [x] Restart cycle: stop the uvicorn on the target port, serve `staging_api:app` on the same port, re-run the identical diagnostic, delete the copy and restart the original module in a `finally`
- [x] Guardrails: refuse to kill a port holder that is not a uvicorn server, refuse to overwrite a pre-existing `staging_api.py`, and never write the measured file
- [x] Before/after comparison on avg/p95 latency, throughput, failed requests and CPU delta, with a 5% noise floor and failures outranking speedups
- [x] `agent.py` asks "Would you like to apply this fix to staging and verify it? (y/n)", only in an interactive terminal; `--verify` and `--no-verify` cover unattended runs
- [x] A failed verification degrades to the diagnosis plus the reason instead of a traceback
- [x] Tests for the rewrite prompt contract, the validation gate, the comparison rules and the cleanup/restore path (`tests/test_phase5_feedback.py`)
- [x] Verify live: `py agent.py --verify "Diagnose http://127.0.0.1:8000/heavy with 12 requests"` staged the model's `ProcessPoolExecutor` rewrite and measured it at 1532 ms -> 3322 ms average latency (a 116.8% increase, improvement -116.8%; p95 1572 ms -> 4787 ms, throughput 3.94 -> 2.44 req/s, CPU delta +9.3 points, 0 failures) on an 8-core host, i.e. **regressed**: 12 concurrent 1.5 s CPU tasks plus pool startup and IPC cost more than the threadpool they replaced. `dummy_api.py` was untouched, `staging_api.py` was deleted, and `dummy_api:app` was serving on port 8000 again afterwards. The loop's first live verdict is a rejection - which is the point: the same fix read as plausible in prose and would have shipped unmeasured.

Deferred past Phase 5:

- [ ] Containerised staging for targets whose dependencies cannot be installed on the host
- [ ] Regression gate across the other endpoints, so a fix cannot trade one bottleneck for another
- [ ] Persist each before/after pair next to the metrics it was computed from

## Phase 6: Remote Audit (Completed)

The first five phases assume the target runs on this machine: they can read its
source, write a fix, and re-measure it. That assumption breaks for a live API,
where the code is not available and a suggested patch cannot be verified. Phase 6
adds the honest alternative: measure the API and report its health, without
pretending to remediate code the agent cannot see.

- [x] Mode selection: loopback targets (`localhost`, `127.0.0.1`, `::1`) keep the Local diagnosis/fix/verify path; every other host uses Audit mode
- [x] Audit mode bypasses `source_file_for`, `load_source_context`, `suggest_code_fix` and `run_feedback_loop` structurally, not by asking the model to skip them
- [x] Dedicated `AUDIT_SYSTEM_PROMPT` with no source-code or remediation stage
- [x] Deterministic verdict in `build_audit_report`: `Broken` if any request failed, `Degraded` if p95 > 2.0 s with no failures, `Stable` otherwise, `Inconclusive` when no load numbers exist
- [x] Model-authored `**Verdict:**` lines are stripped and any `Suggested Fix` section is cut, so the computed verdict cannot be overridden
- [x] If the prompt has no URL, a remote measured URL switches the run to Audit mode via `AUDIT_MODE_NOTICE`
- [x] `--source` and `--verify` are ignored in Audit mode
- [x] Tests for mode detection, verdict thresholds, report composition, model overrides, the measured-URL switch, and the Local-mode regression (`tests/test_phase6_audit.py`)
- [x] Verify live: `py agent.py "Audit https://httpbin.org/status/500 with 5 requests"` measured 5/5 failures at p95 2199 ms and reported `**Verdict:** Broken`, with no source lookup or suggester call

