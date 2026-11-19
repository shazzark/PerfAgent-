"""Remediation suggester - Phase 4: AI Analysis & Suggestion.

Phase 3 answers *why* an endpoint is slow by correlating latency with CPU and
memory. This module turns that diagnosis into something actionable: one narrow
LLM call whose only job is to propose a specific code or infrastructure fix.

Division of responsibility:

* `tools.py` measures and never imports the LLM.
* `suggester.py` never measures; it only asks the model for a fix.
* `agent.py` owns both calls and their order.

The orchestrator invokes `suggest_code_fix(diagnosis)` after the model has
written its analysis, and it is deliberately *not* offered to the model as a
selectable tool. Two reasons: the remediation is then produced exactly once
even when the model forgets to ask for it, and the summary handed over is the
model's own finished diagnosis rather than a half-formed intermediate thought.

Usage:
    py suggester.py            # smoke test on a canned CPU-bound diagnosis
"""

from openai import OpenAI

from llm_client import MODEL, create_client

__all__ = [
    "FIX_PROMPT_TEMPLATE",
    "SUGGESTION_HEADING",
    "SUGGEST_SYSTEM_PROMPT",
    "suggest_code_fix",
]

# Heading the orchestrator puts between the diagnosis and the fix.
SUGGESTION_HEADING = "## Suggested fix"

# The diagnosis is injected verbatim into this contract. The model must return
# something the user can apply, not a prose plan: a diff, a command, an index.
FIX_PROMPT_TEMPLATE = (
    "Based on this bottleneck: {diagnosis_summary}, provide a specific code or "
    "infrastructure fix. Output the fix as a markdown code block (e.g., a Python "
    "diff, a Docker command, or an SQL index). Keep it brief."
)

SUGGEST_SYSTEM_PROMPT = (
    "You are a senior performance engineer writing one remediation for a "
    "measured API bottleneck. "
    "Reply with at most two short sentences naming what the change does, "
    "followed by exactly one fenced markdown code block containing the fix. "
    "Choose the change with the largest measured impact: move CPU-bound work off "
    "the request path (async workers, a process pool), cache repeated work, add "
    "a database index, pool connections, or bound concurrency. "
    "The block must be directly usable - a valid diff, a complete command, or a "
    "statement the user can run. "
    "Do not repeat the diagnosis, do not invent metrics, and do not write tests."
)


def suggest_code_fix(
    diagnosis_summary: str,
    client: OpenAI | None = None,
) -> str:
    """Ask the model for one concrete fix to `diagnosis_summary`.

    Args:
        diagnosis_summary: The orchestrator's finished diagnosis, including the
            measured evidence (latency, CPU/memory deltas, endpoint).
        client: Optional pre-built client, so the orchestrator can reuse its own
            instead of building a second one.

    Returns:
        The model's reply: a short lead-in plus one fenced markdown code block.

    Raises:
        ValueError: If `diagnosis_summary` is empty or not a string.
        RuntimeError: If the model returns an empty message.
    """
    if not isinstance(diagnosis_summary, str) or not diagnosis_summary.strip():
        raise ValueError("diagnosis_summary must be a non-empty string")

    client = client or create_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SUGGEST_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": FIX_PROMPT_TEMPLATE.format(
                    diagnosis_summary=diagnosis_summary.strip()
                ),
            },
        ],
    )

    fix = (response.choices[0].message.content or "").strip()
    if not fix:
        raise RuntimeError("the model returned an empty suggestion")
    return fix


if __name__ == "__main__":
    # Smoke test: a canned diagnosis in the shape `agent.py` produces.
    SAMPLE_DIAGNOSIS = (
        "CPU rose from 24% idle to a 96% peak while 12 requests to /heavy were "
        "in flight, and memory stayed flat (delta -0.3 points). Per-request "
        "latency was 1.59 s against 1.5 s of CPU work per request, so the handler "
        "is CPU-bound: ten workers oversubscribe eight logical CPUs and the "
        "requests queue for cores."
    )
    print(f"🩺 Diagnosis: {SAMPLE_DIAGNOSIS}\n")
    print(suggest_code_fix(SAMPLE_DIAGNOSIS))
