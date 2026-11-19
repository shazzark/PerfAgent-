---
name: Bug report
about: Report something that is broken or behaves unexpectedly
title: "[Bug]: "
labels: ["bug", "triage"]
assignees: []
---

## What happened

A clear description of the problem. One defect per report, please.

## Steps to reproduce

1.
2.
3.

Include the exact command, for example:

```powershell
py agent.py "Diagnose http://127.0.0.1:8000/slow with 10 requests"
```

## Expected behaviour

What you expected PerfAgent to do.

## Actual behaviour

What happened instead. Paste the traceback or the relevant output block, and say
whether it was consistent or intermittent.

```text
<paste output here>
```

## Environment

- OS:
- Python version (`py --version`):
- PerfAgent commit or release:
- Command or prompt used:
- Target: local `dummy_api.py` / remote URL / other
- If a load test was involved: endpoint, request count, and observed latency

## Checks

- [ ] `py -m unittest discover -s tests -t .` passes locally on a clean checkout.
- [ ] I searched existing issues and pull requests for the same problem.
- [ ] I removed every API key, token, `.env` value, and private hostname from
      this report. If a key was exposed, I rotated it before submitting.

## Additional context

Anything else that would help reproduce or diagnose the problem: logs, a
screenshot of the report, CPU/memory context, or a link to a minimal repro.
