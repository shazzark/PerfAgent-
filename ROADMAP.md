# PerfAgent Roadmap

## Phase 1: The Muscle (Tools - Completed)

- [x] Dummy target API with a healthy and a deliberately slow endpoint (`dummy_api.py`)
- [x] Concurrent load-testing tool (`tools.py`, re-exported by `load_test_tool.py`)
- [x] Throughput, latency, and failure reporting verified against the dummy API

## Phase 2: The Brain (Orchestrator - In Progress)

- [x] Connect to Groq through the OpenAI Python SDK
- [x] Expose `run_load_test` to the model as a tool schema
- [x] Execute tool calls and feed results back for analysis
- [ ] Harden the prompt and error handling based on live runs

## Phase 3: The Eyes (Profiler)

- [ ] Instrument the target API to capture CPU, memory, and request traces
- [ ] Correlate database query time with endpoint response latency
- [ ] Persist raw metrics for every load-test run

## Phase 4: AI Analysis

- [ ] Feed profiling data to the LLM for root-cause analysis
- [ ] Rank bottlenecks by measured impact
- [ ] Produce concrete remediation suggestions (indexes, caching, pooling)

## Phase 5: Feedback Loop

- [ ] Apply proposed fixes in a Docker staging environment
- [ ] Re-run the same load test and compare before/after metrics
- [ ] Report a regression-checked summary of what changed and what improved
