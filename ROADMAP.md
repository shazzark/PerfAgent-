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

## Phase 3: The Eyes (Profiler - In Progress)

- [x] Add a CPU-bound `/heavy` endpoint to the dummy target API (time-boxed 256 KB hashing loop, ~1.5 s per request, spread across cores)
- [x] Machine-level CPU and memory profiler (`profiler.py`, `psutil`)
- [x] `run_full_diagnostic` runs the load test on a background thread while the profiler samples the same window, returning one combined payload
- [x] Update the agent tool schema and system prompt so latency is correlated with CPU and memory
- [ ] Instrument the target API to capture per-request CPU and request traces
- [ ] Correlate database query time with endpoint response latency
- [ ] Persist raw metrics for every load-test run

Current Phase 3 limitation: the profiler measures the whole machine, not the
target process, so a load generator sharing the host contributes to the CPU
figure. The fixed 5 s sampling window can also be shorter than a long load test;
`profile_covers_full_load_test` reports when that happened.


## Phase 4: AI Analysis

- [ ] Feed profiling data to the LLM for root-cause analysis
- [ ] Rank bottlenecks by measured impact
- [ ] Produce concrete remediation suggestions (indexes, caching, pooling)

## Phase 5: Feedback Loop

- [ ] Apply proposed fixes in a Docker staging environment
- [ ] Re-run the same load test and compare before/after metrics
- [ ] Report a regression-checked summary of what changed and what improved
