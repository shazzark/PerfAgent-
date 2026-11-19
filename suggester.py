"""Remediation suggester - Phase 4: AI Analysis & Suggestion.

Phase 3 answers *why* an endpoint is slow by correlating latency with CPU and
memory. This module turns that diagnosis into something actionable: one narrow
LLM call whose only job is to propose a specific code or infrastructure fix.

Phase 4.5 closes the remaining hole in that call. A diagnosis alone says nothing
about *what the fix is written in*: the first live run guessed Flask, invented
`app/routes.py`, and patched a `Blueprint` that does not exist, because the
target is a small FastAPI app. So the target's source code is now a required
input rather than optional context. `source_code_context` has no default on
purpose - a caller cannot accidentally buy a framework hallucination by
forgetting it.

Division of responsibility:

* `tools.py` measures and never imports the LLM.
* `suggester.py` never measures and never reads files; it only asks the model
  for a fix against the code it is handed.
* `agent.py` owns both calls and their order, including reading the target file.

The orchestrator invokes `suggest_code_fix(diagnosis, source_code_context)`
after the model has written its analysis, and it is deliberately *not* offered
to the model as a selectable tool. Two reasons: the remediation is then
produced exactly once even when the model forgets to ask for it, and the summary
handed over is the model's own finished diagnosis rather than a half-formed
intermediate thought.

Usage:
    py suggester.py            # smoke test on a canned CPU-bound diagnosis
"""

import re

from openai import OpenAI

from llm_client import MODEL, create_client

__all__ = [
    "FIX_PROMPT_TEMPLATE",
    "NO_SOURCE_CONTEXT",
    "STAGED_FILE_PROMPT_TEMPLATE",
    "STAGED_FILE_SYSTEM_PROMPT",
    "SUGGESTION_HEADING",
    "SUGGEST_SYSTEM_PROMPT",
    "extract_python_source",
    "suggest_code_fix",
    "suggest_staged_file",
]

# Heading the orchestrator puts between the diagnosis and the fix.
SUGGESTION_HEADING = "## Suggested fix"

# Substituted when the caller could not read the target's code. It is stated
# explicitly so the model knows the framework is unknown instead of silently
# reverting to the Flask-shaped guess this parameter exists to prevent.
NO_SOURCE_CONTEXT = (
    "(the target's source code could not be read, so its framework and handler "
    "signatures are unknown)"
)

# The diagnosis and the target's code are injected verbatim into this contract.
# The model must return something the user can apply, not a prose plan, and it
# must match the code it was shown rather than a framework it expects to find.
FIX_PROMPT_TEMPLATE = (
    "Based on this bottleneck: {diagnosis_summary}, provide a specific code or "
    "infrastructure fix for the application whose source code is below.\n\n"
    "Target source code:\n"
    "```\n{source_code_context}\n```\n\n"
    "Write the change in the language and framework this code actually uses: "
    "reuse its imports, decorators, route paths and handler signatures, and do "
    "not replace one framework with another or reference files the code does not "
    "show. Output the fix as a markdown code block (e.g., a Python diff, a Docker "
    "command, or an SQL index). Keep it brief."
)

SUGGEST_SYSTEM_PROMPT = (
    "You are a senior performance engineer writing one remediation for a "
    "measured API bottleneck. "
    "The user message includes the target application's source code: read it, "
    "and make the fix fit that exact codebase - same language, same framework, "
    "same imports and handler style. "
    "Never substitute a different framework, never invent files or routes the "
    "source does not contain, and if the source is missing say the fix is "
    "framework-agnostic instead of guessing one. "
    "Reply with at most two short sentences naming what the change does, "
    "followed by exactly one fenced markdown code block containing the fix. "
    "Choose the change with the largest measured impact: move CPU-bound work off "
    "the request path (async workers, a process pool), cache repeated work, add "
    "a database index, pool connections, or bound concurrency. "
    "The block must be directly usable - a valid diff, a complete command, or a "
    "statement the user can run. "
    "Do not repeat the diagnosis, do not invent metrics, and do not write tests."
)


# Phase 5 needs a runnable artifact, not a patch. LLM-written diffs are brittle to
# apply (context drift, wrong line counts, invented hunks), so the staging loop
# asks for the whole file back and writes it over a copy. That trades prompt size
# for something a parser-free applier can trust: the only validation needed is
# "does this compile and still serve the route".
STAGED_FILE_PROMPT_TEMPLATE = (
    "Rewrite the complete application file below with this fix applied.\n\n"
    "Fix:\n{suggested_fix}\n\n"
    "Current file ({file_name}):\n"
    "```python\n{source_code}\n```\n\n"
    "Return the ENTIRE modified file - not a diff, not a fragment, not a "
    "summary. Every import, route and helper the fix does not need to change must "
    "stay exactly as it is, and the file must keep serving the same routes. "
    "Output exactly one ```python code block and nothing else."
)

STAGED_FILE_SYSTEM_PROMPT = (
    "You are a senior Python engineer applying one already-decided performance "
    "fix to one file. "
    "You return the whole file, ready to run, with no commentary and no "
    "truncation placeholders such as '# ... unchanged ...'. "
    "Never return a diff, never rename the app object, the routes, or the helper "
    "functions the existing code already uses, and never introduce imports that "
    "are not used. "
    "If the fix cannot be applied safely to this file, return the file unchanged "
    "rather than inventing a new framework or a new module layout."
)

# Matches ```python ... ```, ```py ... ``` and bare ``` ... ``` fences. Models
# sometimes ignore the "code block only" instruction and add a sentence, so the
# fenced body is extracted instead of trusting the whole reply.
_FENCE_RE = re.compile(
    r"```[ \t]*(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE
)


def extract_python_source(reply: str) -> str:
    """Return the Python source contained in a model reply.

    Args:
        reply: The raw reply, either a fenced code block or bare source.

    Returns:
        The block content, newline-terminated, so it can be written to disk.

    Raises:
        ValueError: If the reply is empty or not a string.
    """
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("the model returned no file content")

    match = _FENCE_RE.search(reply)
    source = match.group(1) if match else reply
    source = source.strip("\n")
    if not source.strip():
        raise ValueError("the model returned an empty code block")
    return source + "\n"


def suggest_code_fix(
    diagnosis_summary: str,
    source_code_context: str,
    client: OpenAI | None = None,
) -> str:
    """Ask the model for one concrete fix to `diagnosis_summary`.

    Args:
        diagnosis_summary: The orchestrator's finished diagnosis, including the
            measured evidence (latency, CPU/memory deltas, endpoint).
        source_code_context: The target's source code, as read from disk. A blank
            string is accepted and replaced with `NO_SOURCE_CONTEXT`, so a caller
            that could not read the file still gets a suggestion - merely a
            framework-agnostic one.
        client: Optional pre-built client, so the orchestrator can reuse its own
            instead of building a second one.

    Returns:
        The model's reply: a short lead-in plus one fenced markdown code block.

    Raises:
        ValueError: If `diagnosis_summary` is empty or not a string, or if
            `source_code_context` is not a string.
        RuntimeError: If the model returns an empty message.
    """
    if not isinstance(diagnosis_summary, str) or not diagnosis_summary.strip():
        raise ValueError("diagnosis_summary must be a non-empty string")
    if not isinstance(source_code_context, str):
        raise ValueError("source_code_context must be a string")

    context = source_code_context.strip() or NO_SOURCE_CONTEXT

    client = client or create_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SUGGEST_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": FIX_PROMPT_TEMPLATE.format(
                    diagnosis_summary=diagnosis_summary.strip(),
                    source_code_context=context,
                ),
            },
        ],
    )

    fix = (response.choices[0].message.content or "").strip()
    if not fix:
        raise RuntimeError("the model returned an empty suggestion")
    return fix


def suggest_staged_file(
    source_code: str,
    suggested_fix: str,
    file_name: str = "app.py",
    client: OpenAI | None = None,
) -> str:
    """Ask the model to apply `suggested_fix` and return the complete file.

    Used by the Phase 5 staging loop, which cannot safely apply a model-written
    diff. The reply is reduced to Python source by `extract_python_source`; the
    caller is still responsible for compiling and validating it before running it.

    Args:
        source_code: The complete current contents of the file.
        suggested_fix: The remediation produced by `suggest_code_fix`, as-is.
        file_name: Name shown to the model, so it does not invent a path.
        client: Optional pre-built client, reused from the orchestrator.

    Returns:
        The rewritten file, newline-terminated.

    Raises:
        ValueError: If either input is empty or not a string.
        RuntimeError: If the model returns an empty message.
    """
    if not isinstance(source_code, str) or not source_code.strip():
        raise ValueError("source_code must be a non-empty string")
    if not isinstance(suggested_fix, str) or not suggested_fix.strip():
        raise ValueError("suggested_fix must be a non-empty string")

    client = client or create_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": STAGED_FILE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": STAGED_FILE_PROMPT_TEMPLATE.format(
                    suggested_fix=suggested_fix.strip(),
                    file_name=file_name,
                    source_code=source_code.strip(),
                ),
            },
        ],
    )

    reply = (response.choices[0].message.content or "").strip()
    if not reply:
        raise RuntimeError("the model returned an empty staged file")
    return extract_python_source(reply)


if __name__ == "__main__":
    # Smoke test: a canned diagnosis in the shape `agent.py` produces, against a
    # minimal FastAPI app so the framework is unambiguous in the reply.
    SAMPLE_DIAGNOSIS = (
        "CPU rose from 24% idle to a 96% peak while 12 requests to /heavy were "
        "in flight, and memory stayed flat (delta -0.3 points). Per-request "
        "latency was 1.59 s against 1.5 s of CPU work per request, so the handler "
        "is CPU-bound: ten workers oversubscribe eight logical CPUs and the "
        "requests queue for cores."
    )
    SAMPLE_SOURCE = (
        "from fastapi import FastAPI\n"
        "\n"
        "app = FastAPI()\n"
        "\n"
        "@app.get(\"/heavy\")\n"
        "def heavy():\n"
        "    return {\"iterations\": _burn_cpu(1.5)}\n"
    )
    print(f"🩺 Diagnosis: {SAMPLE_DIAGNOSIS}\n")
    print(f"📄 Target source:\n{SAMPLE_SOURCE}\n")
    print(suggest_code_fix(SAMPLE_DIAGNOSIS, SAMPLE_SOURCE))
