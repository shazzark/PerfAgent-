"""PerfAgent orchestrator - Phase 2: The Brain.

The orchestrator is one LLM tool-calling loop:

1. Send the user's goal, plus the `run_load_test` tool schema, to Groq.
2. If the model asks for the tool, execute it locally and feed the JSON result
   back as a `tool` message.
3. Repeat until the model answers without requesting another tool.

Groq exposes an OpenAI-compatible API, so the official `openai` SDK is used
with `base_url` pointing at `https://api.groq.com/openai/v1`.

Usage:
    py agent.py "Load test http://127.0.0.1:8000/slow with 20 requests"
    py agent.py            # runs the built-in demo prompt
"""

import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from tools import run_load_test

load_dotenv()

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Safety valve: a well-behaved model needs one round to call the tool and one
# to explain the result. Anything beyond this is a loop and should fail loudly.
MAX_TOOL_ROUNDS = 5

DEFAULT_PROMPT = (
    "Please load test the endpoint http://127.0.0.1:8000/slow with 20 requests. "
    "Tell me if it is fast or slow, and what the bottleneck likely is."
)

SYSTEM_PROMPT = (
    "You are PerfAgent, an autonomous API performance engineer. "
    "Whenever the user asks you to measure, test, or benchmark an endpoint, "
    "you must call the run_load_test tool instead of estimating results yourself. "
    "Never invent latency, throughput, or failure numbers. "
    "If the target URL is missing, ask the user for it. "
    "After a test, report throughput, success and failure counts, then name the "
    "most likely bottleneck using only the returned metrics and suggest the next "
    "diagnostic step. Keep the answer concise and technical."
)

BASE_URL = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
MODEL = os.getenv("LLM_MODEL", DEFAULT_MODEL)

# The model reads this schema to decide whether and how to call the tool.
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_load_test",
            "description": (
                "Run a concurrent HTTP load test against a single endpoint and "
                "return latency, success/failure counts, and requests per second."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "Absolute URL to test, including scheme and port "
                            "(for example http://127.0.0.1:8000/slow)."
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


def create_client() -> OpenAI:
    """Build the Groq-backed OpenAI client, failing early on missing config."""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is not set. Put it in .env (see README.md, 'Configuration')."
        )
    return OpenAI(api_key=api_key, base_url=BASE_URL)


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
    if name != "run_load_test":
        return {"error": f"Unknown tool: {name}"}

    url = arguments.get("url")
    if not isinstance(url, str) or not url:
        return {"error": "run_load_test requires a non-empty 'url' argument"}

    try:
        total_requests = int(arguments.get("total_requests", 10))
        return run_load_test(url=url, total_requests=total_requests)
    except (TypeError, ValueError) as exc:
        return {"error": f"Invalid arguments for run_load_test: {exc}"}


def run_agent(user_prompt: str, client: OpenAI | None = None) -> str:
    """Resolve `user_prompt` with tool calls and return the model's final answer."""
    client = client or create_client()
    print(f"👤 User: {user_prompt}\n")

    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for _round in range(1, MAX_TOOL_ROUNDS + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        if not message.tool_calls:
            answer = message.content or "(the model returned an empty response)"
            print(f"🤖 PerfAgent: {answer}")
            return answer

        # The assistant turn must be echoed back before its tool results,
        # otherwise the next request is rejected for an orphaned tool_call_id.
        messages.append(message)

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = _parse_arguments(tool_call.function.arguments)
            print(f"🛠️  Calling {name} with {arguments}")

            result = _execute_tool(name, arguments)
            print(f"   ↳ Result: {json.dumps(result)}")

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


