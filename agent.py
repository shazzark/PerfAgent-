"""PerfAgent orchestrator - Phase 4: AI Analysis & Suggestion.

The orchestrator is one LLM tool-calling loop followed by one remediation call:

1. Send the user's goal, plus the `run_full_diagnostic` tool schema, to Groq.
2. If the model asks for the tool, execute it locally (load test + CPU/memory
   profiling) and feed the JSON result back as a `tool` message.
3. Repeat until the model answers without requesting another tool. That answer
   is the diagnosis.
4. If a diagnostic actually ran, resolve the file that declares the measured
   route, read it, and call `suggester.suggest_code_fix(diagnosis, source)`, so
   the fix is written against the real framework instead of a guess. The answer
   is the diagnosis plus one concrete code/infrastructure fix.

Groq exposes an OpenAI-compatible API, so the official `openai` SDK is used
with `base_url` pointing at `https://api.groq.com/openai/v1`.

Usage:
    py agent.py "Diagnose http://127.0.0.1:8000/heavy with 12 requests"
    py agent.py --source path/to/app.py "Diagnose http://host:8000/heavy"
    py agent.py            # runs the built-in demo prompt
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI

from llm_client import MODEL, create_client
from suggester import SUGGESTION_HEADING, suggest_code_fix
from tools import run_full_diagnostic

# Safety valve: a well-behaved model needs one round to call the tool and one
# to explain the result. Anything beyond this is a loop and should fail loudly.
MAX_TOOL_ROUNDS = 5

# The remediation call is only trustworthy when the model can see the code it is
# patching: without context a live run guessed Flask for a FastAPI service and
# produced a diff against files that do not exist. `run_agent` resolves the file
# from (in order) an explicit argument, this environment variable, or the project
# file that declares the measured route.
SOURCE_FILE_ENV_VAR = "PERFAGENT_SOURCE_FILE"

# The agent's own modules are never the system under test, so the route scan
# skips them: `agent.py` and friends mention endpoint paths in docstrings.
_AGENT_FILES = frozenset(
    {
        "agent.py",
        "tools.py",
        "suggester.py",
        "profiler.py",
        "llm_client.py",
        "load_test_tool.py",
        "list_models.py",
        "test_connection.py",
    }
)

# A whole module is usually what the fix needs; capping it keeps the remediation
# prompt from being dominated by a large file the model cannot use anyway.
MAX_SOURCE_CONTEXT_CHARS = 16_000

_PROJECT_ROOT = Path(__file__).resolve().parent

# Matches `@app.get("/heavy")`, `@router.post('/jobs')`, `@bp.route("/x", ...)`
# and the FastAPI/Starlette aliases, with the path captured exactly so a
# docstring mention of "/heavy" is not mistaken for a declaration.
_ROUTE_DECORATOR_TEMPLATE = (
    r"@[\w.]+\.(?:get|post|put|patch|delete|head|options|route|api_route)\("
    r"\s*[\"']{path}[\"']"
)

DEFAULT_PROMPT = (
    "Run a full diagnostic on http://127.0.0.1:8000/heavy with 12 requests. "
    "Correlate the CPU usage with the response times and tell me whether the "
    "bottleneck is CPU-bound or waiting on something else."
)

SYSTEM_PROMPT = (
    "You are PerfAgent, an autonomous API performance engineer. "
    "Whenever the user asks you to measure, test, benchmark, or diagnose an "
    "endpoint, you must call the run_full_diagnostic tool instead of estimating "
    "results yourself. "
    "That tool load-tests the endpoint while sampling machine CPU and memory, and "
    "it also samples a two second idle baseline immediately before the test. "
    "Never invent latency, throughput, CPU, memory, or failure numbers. "
    "If the target URL is missing, ask the user for it. "
    "When you read the result, follow these rules:\n"
    "1. Absolute CPU and memory are machine-wide: they include the load "
    "generator, the agent process, and any unrelated software on the host. Never "
    "call a workload memory-bound because the absolute memory number is high. A "
    "resource only implicates the target if it *changed* while the requests were "
    "in flight, which is what cpu_delta_percent and memory_delta_percent measure, "
    "and what the cpu_samples series shows second by second.\n"
    "2. Quote the baseline next to the load figures, for example 'CPU rose from "
    "24% idle to 43% average (peak 99%) during the run, a delta of +19 points'.\n"
    "3. An average is diluted when the whole load test fits in a fraction of the "
    "five second window, so weigh peak_cpu_percent and the cpu_samples spike "
    "alongside the average. profile_covers_full_load_test=false means the metrics "
    "describe only a prefix of the run.\n"
    "4. Per-request latency is in load_test.avg_latency_seconds, "
    "p95_latency_seconds and max_latency_seconds. Never derive latency by dividing "
    "total_time_seconds by total_requests: requests run ten at a time, so that "
    "division understates what a single client waited by roughly the concurrency "
    "factor.\n"
    "5. Then name the most likely bottleneck: CPU rising sharply with per-request "
    "latency near the CPU work time indicates a CPU-bound workload; high latency "
    "with flat CPU and memory deltas indicates sleeping, blocking I/O, or a "
    "downstream dependency; failures or timeouts point at saturation or a "
    "connection limit.\n"
    "Report throughput, success and failure counts, the CPU and memory deltas, "
    "and how confident the reading is given the sampling window. Name the single "
    "most likely bottleneck first and rank any others by measured impact. Keep "
    "the answer concise and technical. Do not write code, a patch, or a "
    "remediation plan: a separate step turns this diagnosis into one concrete "
    "fix, so the diagnosis is all that is needed here."
)

# The model reads this schema to decide whether and how to call the tool.
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_full_diagnostic",
            "description": (
                "Run a concurrent HTTP load test against a single endpoint while "
                "profiling machine CPU and memory over a 5 second window. Samples a "
                "2 second idle baseline immediately before the test, then returns "
                "latency (average, p95, max per request), success/failure counts and "
                "requests per second, plus "
                "average and peak CPU/memory percentages, the per-second sample "
                "series, and the idle-to-load deltas (cpu_delta_percent, "
                "memory_delta_percent) for the same run. Use the deltas and the "
                "peak rather than raw absolute values, because the absolute figures "
                "include unrelated software on the host."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "Absolute URL to test, including scheme and port "
                            "(for example http://127.0.0.1:8000/heavy)."
                        ),
                    },
                    "total_requests": {
                        "type": "integer",
                        "description": "Number of requests to send. Minimum 1.",
                        "minimum": 1,
                    },
                },
                "required": ["url", "total_requests"],
                "additionalProperties": False,
            },
        },
    }
]



def _parse_arguments(raw_arguments: str) -> dict[str, Any]:
    """Decode the model's JSON argument string, tolerating malformed output."""
    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run a tool by name and return a JSON-serialisable result.

    Tool failures are returned as data rather than raised so the model can
    correct its arguments or report the problem instead of crashing the run.
    """
    if name != "run_full_diagnostic":
        return {"error": f"Unknown tool: {name}"}

    url = arguments.get("url")
    if not isinstance(url, str) or not url:
        return {"error": "run_full_diagnostic requires a non-empty 'url' argument"}

    try:
        total_requests = int(arguments.get("total_requests", 10))
        return run_full_diagnostic(url=url, total_requests=total_requests)
    except (TypeError, ValueError) as exc:
        return {"error": f"Invalid arguments for run_full_diagnostic: {exc}"}



def _endpoint_path(url: str | None) -> str | None:
    """Return the request path of `url`, or None when there is no usable URL."""
    if not isinstance(url, str) or not url.strip():
        return None
    return urlparse(url).path or "/"


def find_source_file(url: str | None, root: Path = _PROJECT_ROOT) -> Path | None:
    """Return the project file that declares `url`'s route, if one exists.

    Deliberately narrow: only `*.py` files in the project root are read, and only
    an exact decorator path counts, so prose about "/heavy" cannot masquerade as
    a definition. A richer index (AST, router registry) would encode assumptions
    about other people's layouts; an unresolvable endpoint simply means the
    suggestion is written without context.
    """
    path = _endpoint_path(url)
    if path is None:
        return None

    pattern = re.compile(_ROUTE_DECORATOR_TEMPLATE.format(path=re.escape(path)))
    for candidate in sorted(root.glob("*.py")):
        if candidate.name in _AGENT_FILES:
            continue
        try:
            source = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if pattern.search(source):
            return candidate
    return None


def _resolve_source_file(
    source_file: str | Path | None, url: str | None
) -> Path | None:
    """Pick the file whose code the suggester should see.

    Precedence is explicit argument, then `PERFAGENT_SOURCE_FILE`, then the file
    that declares the measured route. A configured path that does not exist is
    reported as missing rather than silently replaced by a guess.
    """
    configured = source_file or os.getenv(SOURCE_FILE_ENV_VAR)
    if not configured:
        return find_source_file(url)

    path = Path(configured).expanduser()
    if path.is_absolute():
        return path if path.is_file() else None

    for base in (_PROJECT_ROOT, Path.cwd()):
        candidate = base / path
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_source_context(
    url: str | None, source_file: str | Path | None = None
) -> str:
    """Read the target's source so the fix is written against the real code.

    Returns an empty string when no file can be resolved or read. That is not
    fatal: losing a completed diagnosis to a missing file would be the worse
    outcome, and `suggester` turns the empty string into an explicit
    framework-unknown note instead of guessing. The reason is printed so the
    user knows why a fix may be generic.
    """
    path = _resolve_source_file(source_file, url)
    if path is None:
        print(
            "📄 No source file matched the measured endpoint; "
            "suggesting without code context."
        )
        return ""

    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"⚠️  Could not read {path}: {exc}")
        return ""

    if len(source) > MAX_SOURCE_CONTEXT_CHARS:
        source = (
            source[:MAX_SOURCE_CONTEXT_CHARS]
            + f"\n... [source truncated at {MAX_SOURCE_CONTEXT_CHARS} characters] ..."
        )

    source = source.strip()
    if not source:
        print(f"📄 {path.name} is empty; suggesting without code context.")
        return ""

    print(f"📄 Attaching {path.name} ({len(source)} chars) as source code context.")
    # The filename header lets the model emit a diff that names the real file
    # instead of inventing `app/routes.py`.
    return f"# File: {path.name}\n{source}"


def _with_suggested_fix(
    diagnosis: str,
    client: OpenAI,
    url: str | None = None,
    source_file: str | Path | None = None,
) -> str:
    """Return `diagnosis` followed by one concrete remediation.

    The suggestion is a second, narrowly scoped model call that now also receives
    the target's source code, which is what keeps the fix on the same framework
    the service is actually built on. A provider failure here must not discard a
    completed measurement, so the error is reported in the answer itself instead
    of being raised.
    """
    source_context = load_source_context(url, source_file)
    print("🩺 Asking the model for a concrete fix...")
    try:
        fix = suggest_code_fix(diagnosis, source_context, client=client)
    except Exception as exc:  # noqa: BLE001 - keep the diagnosis, surface the cause
        print(f"⚠️  Suggestion step failed: {exc}")
        return (
            f"**Diagnosis**\n\n{diagnosis}\n\n{SUGGESTION_HEADING}\n\n"
            f"_Suggestion unavailable: {exc}_"
        )

    report = f"**Diagnosis**\n\n{diagnosis}\n\n{SUGGESTION_HEADING}\n\n{fix}"
    return report


def run_agent(
    user_prompt: str,
    client: OpenAI | None = None,
    source_file: str | Path | None = None,
) -> str:
    """Resolve `user_prompt`, then turn the measured diagnosis into a fix.

    The tool loop yields the diagnosis; the remediation call that follows is
    what makes the run actionable. `diagnostic_ran` gates that second call, so a
    purely conversational answer is returned unchanged instead of being sent off
    for an unrelated patch, and the URL the tool actually measured is what seeds
    the source-file lookup, so the fix is written against the file that was
    really exercised. `source_file` overrides that lookup when the caller knows
    the target.
    """
    client = client or create_client()
    print(f"👤 User: {user_prompt}\n")

    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    diagnostic_ran = False
    diagnostic_url: str | None = None

    for _round in range(1, MAX_TOOL_ROUNDS + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        if not message.tool_calls:
            analysis = (message.content or "").strip()
            if not analysis:
                answer = "(the model returned an empty response)"
                print(f"🤖 PerfAgent: {answer}")
                return answer
            if not diagnostic_ran:
                # Nothing was measured, so there is no bottleneck to remediate.
                print(f"🤖 PerfAgent: {analysis}")
                return analysis

            report = _with_suggested_fix(analysis, client, diagnostic_url, source_file)
            print(f"🤖 PerfAgent (diagnosis + suggested fix):\n{report}")
            return report

        # The assistant turn must be echoed back before its tool results,
        # otherwise the next request is rejected for an orphaned tool_call_id.
        messages.append(message)

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = _parse_arguments(tool_call.function.arguments)
            print(f"🛠️  Calling {name} with {arguments}")

            result = _execute_tool(name, arguments)
            print(f"   ↳ Result: {json.dumps(result)}")
            # Only a diagnostic that actually produced load numbers justifies a
            # remediation; a failed run has no measured bottleneck to explain.
            if name == "run_full_diagnostic" and result.get("load_test"):
                diagnostic_ran = True
                diagnostic_url = result.get("url") or arguments.get("url")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": json.dumps(result),
                }
            )

    raise RuntimeError(
        f"Model requested tools on {MAX_TOOL_ROUNDS} consecutive rounds without "
        "producing an answer; aborting to avoid an unbounded loop."
    )


if __name__ == "__main__":
    argv = sys.argv[1:]
    source_file: str | None = None
    if "--source" in argv:
        index = argv.index("--source")
        try:
            source_file = argv[index + 1]
        except IndexError:
            print("❌ --source requires a file path")
            raise SystemExit(2) from None
        del argv[index : index + 2]

    prompt = " ".join(argv).strip() or DEFAULT_PROMPT
    # The target API must be running, e.g.:
    #   py -m uvicorn dummy_api:app --reload
    try:
        run_agent(prompt, source_file=source_file)
    except Exception as exc:  # CLI boundary: report the failure, exit non-zero
        print(f"❌ {exc}")
        raise SystemExit(1) from exc


