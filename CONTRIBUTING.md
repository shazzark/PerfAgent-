# Contributing to PerfAgent

Thanks for taking the time to contribute. PerfAgent is a small, test-driven
project, so the fastest way to get a change merged is to keep it focused and to
show the command you ran to verify it.

This guide covers local setup, the test command, and the pull request process.
By participating you agree to keep discussions technical and respectful.

## Before you start

- Open an issue first for anything larger than a typo or a one-line fix, so the
  approach can be agreed before you spend time on it. Use the
  [bug report](.github/ISSUE_TEMPLATE/bug_report.md) or
  [feature request](.github/ISSUE_TEMPLATE/feature_request.md) template.
- Do not report a security problem in a public issue. If a key or a credential
  is exposed, rotate it in the provider console first, then describe the problem
  without the secret. See the Security notes in [README.md](README.md).
- Search existing issues and pull requests before opening a new one. Duplicate
  reports are closed with a link, not folded in silently.

## Local setup

Requirements: Python 3.13, a Groq API key (the free tier works), and Windows,
macOS, or Linux.

```powershell
# 1. Clone your fork and enter the project root
git clone https://github.com/<your-user>/PerfAgent-.git
Set-Location -LiteralPath .\PerfAgent-

# 2. Install dependencies
py -m pip install openai python-dotenv fastapi uvicorn requests psutil

# 3. Create .env in the project root (it is gitignored - never commit it)
@"
LLM_API_KEY=your_groq_api_key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
"@ | Set-Content -LiteralPath .env
```

`LLM_BASE_URL` and `LLM_MODEL` are optional; they default to Groq and
`openai/gpt-oss-120b`. On macOS or Linux, replace the here-string with your
shell's heredoc or edit `.env` by hand.

Run the bundled target API in one terminal and the agent in another:

```powershell
py -m uvicorn dummy_api:app --reload
py agent.py "Diagnose http://127.0.0.1:8000/heavy with 10 requests"
```

`py tools.py` exercises the tools without an LLM, which is useful while
iterating on the measurement code.

## Running the tests

The suite is hermetic: it mocks the LLM client and the network, so no API key or
running server is required.

```powershell
py -m unittest discover -s tests -t .
```

Run it from the project root; `-t .` sets the top-level directory so the
`tests` package imports correctly. The suite currently has 85 tests.

A single module during development:

```powershell
py -m unittest tests.test_phase7_ramp -v
```

Also compile the files you touched, which catches syntax errors the tests may
not import:

```powershell
py -m py_compile tools.py agent.py tests/test_phase7_ramp.py
```

If a change affects timing or load behaviour, include the measured output in the
pull request rather than a claim. Note the machine, the endpoint, and the exact
command; single-run numbers on a shared host are evidence, not proof.

## Making a change

1. Create a branch from `main` with a descriptive name, for example
   `feature/k6-generator` or `fix/ramp-steady-state`.
2. Make the smallest change that solves the problem. Do not bundle unrelated
   refactors, formatting churn, or dependency upgrades into a bug fix.
3. Add or update a regression test in `tests/`. Follow the existing
   `test_phaseN_*.py` naming and style: `unittest`, `unittest.mock.patch`, and
   explicit assertions on the returned payload or printed report.
4. Update the documentation the change touches: `README.md` for user-facing
   behaviour and `ARCHITECTURE.md` for contracts, thresholds, payload shapes, or
   design decisions. A behaviour change without a docs update is incomplete.
5. Keep the existing conventions: standard library plus the dependencies already
   in `README.md`, type hints on public functions, docstrings that explain *why*,
   plain-text console prefixes (no emoji), and no new dependency without a
   discussion in an issue first.

Commit messages follow the project's history:

```text
phase-N: short description     # a phase deliverable
fix: short description         # a bug fix
docs: short description        # documentation only
```

## Submitting a pull request

1. Fork the repository, push your branch, and open a pull request against
   `main` on [shazzark/PerfAgent-](https://github.com/shazzark/PerfAgent-).
2. Fill in the pull request description with the problem, the change, the
   verification commands and their output, and any known limitations. Link the
   issue the pull request closes.
3. Confirm the checklist below before requesting review.
4. Respond to review comments with evidence. If you disagree, explain the
   measurement or reasoning; "it works on my machine" is not a resolution.

### Pull request checklist

- [ ] `py -m unittest discover -s tests -t .` passes locally.
- [ ] `py -m py_compile` passes for every changed Python file.
- [ ] A test covers the new behaviour or the fixed defect.
- [ ] `README.md` / `ARCHITECTURE.md` are updated when behaviour or contracts
      change.
- [ ] No `.env`, API key, credential, `staging_api.py`, `__pycache__`, or
      generated `*.json` report is committed.
- [ ] The change is focused; unrelated edits are split into separate pull
      requests.

Maintainers aim to review pull requests within a few days. A pull request that
is green, focused, and documented gets merged; one that changes behaviour
without a test or a measurement is sent back with an explanation.
