"""PerfAgent orchestrator - Phase 4: AI Analysis & Suggestion.

The orchestrator is one LLM tool-calling loop followed by one remediation call:

1. Send the user's goal, plus the `run_full_diagnostic` tool schema, to Groq.
2. If the model asks for the tool, execute it locally (load test + CPU/memory
   profiling) and feed the JSON result back as a `tool` message.
3. Repeat until the model answers without requesting another tool. That answer
   is the diagnosis.
4. If a diagnostic actually ran, call `suggester.suggest_code_fix(diagnosis)` and
   return the diagnosis plus one concrete code/infrastructure fix.

Groq exposes an OpenAI-compatible API, so the official `openai` SDK is used
with `base_url` pointing at `https://api.groq.com/openai/v1`.

Usage:
    py agent.py "Diagnose http://127.0.0.1:8000/heavy with 12 requests"
    py agent.py            # runs the built-in demo prompt
"""

import json
import sys
from typing import Any

from openai import OpenAI

from llm_client import MODEL, create_client
from suggester import SUGGESTION_HEADING, suggest_code_fix
from tools import run_full_diagnostic

# Safety valve: a well-behaved model needs one round to call the tool and one
# to explain the result. Anything beyond this is a loop and should fail loudly.
MAX_TOOL_ROUNDS = 5

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



def _with_suggested_fix(diagnosis: str, client: OpenAI) -> str:
    """Return `diagnosis` followed by one concrete remediation.

    The suggestion is a second, narrowly scoped model call. A provider failure
    here must not discard a completed measurement, so the error is reported in
    the answer itself instead of being raised.
    """
    print("🩺 Asking the model for a concrete fix...")
    try:
        fix = suggest_code_fix(diagnosis, client=client)
    except Exception as exc:  # noqa: BLE001 - keep the diagnosis, surface the cause
        print(f"⚠️  Suggestion step failed: {exc}")
        return (
            f"**Diagnosis**\n\n{diagnosis}\n\n{SUGGESTION_HEADING}\n\n"
            f"_Suggestion unavailable: {exc}_"
        )

    report = f"**Diagnosis**\n\n{diagnosis}\n\n{SUGGESTION_HEADING}\n\n{fix}"
    return report


def run_agent(user_prompt: str, client: OpenAI | None = None) -> str:
    """Resolve `user_prompt`, then turn the measured diagnosis into a fix.

    The tool loop yields the diagnosis; the remediation call that follows is
    what makes the run actionable. `diagnostic_ran` gates that second call, so a
    purely conversational answer is returned unchanged instead of being sent off
    for an unrelated patch.
    """
    client = client or create_client()
    print(f"👤 User: {user_prompt}\n")

    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    diagnostic_ran = False

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

            report = _with_suggested_fix(analysis, client)
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
    prompt = " ".join(sys.argv[1:]).strip() or DEFAULT_PROMPT
    # The target API must be running, e.g.:
    #   py -m uvicorn dummy_api:app --reload
    try:
        run_agent(prompt)
    except Exception as exc:  # CLI boundary: report the failure, exit non-zero
        print(f"❌ {exc}")
        raise SystemExit(1) from exc


