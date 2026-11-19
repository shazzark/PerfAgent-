---
name: Feature request
about: Suggest an improvement or a new capability for PerfAgent
title: "[Feature]: "
labels: ["enhancement"]
assignees: []
---

## Problem

What are you trying to do that PerfAgent makes difficult or impossible today?
Describe the use case rather than jumping straight to an implementation.

## Proposed solution

What you would like to happen. If it changes a tool, a payload field, a prompt,
or a report, name it (for example `find_breaking_point`, `crash_reason`, the
ramp stop rules, or the Audit verdict).

## Alternatives considered

Other tools, flags, or workflows you tried, and why they did not fit.

## Scope

Which area would this touch?

- [ ] Load generator / measurement (`tools.py`, `profiler.py`)
- [ ] Orchestration / prompts / verdicts (`agent.py`, `llm_client.py`)
- [ ] Suggester / staging / comparison (`suggester.py`, `feedback_loop.py`)
- [ ] Documentation (`README.md`, `ARCHITECTURE.md`, `ROADMAP.md`)
- [ ] Tests / developer experience
- [ ] Other:

## Does this need a new dependency?

PerfAgent deliberately keeps its dependency list small. If the proposal needs a
new package, an external service, or a non-Python component (for example k6 for
very high request rates), say so and explain why the existing stack cannot do it.

## Additional context

Prior art, benchmarks, links, or a sketch of the expected output. Note that
requests must not include performance claims without a measurement; if you have
numbers, include the command and the environment they came from.
