"""PerfAgent orchestrator - analysis, remote Audit mode, and ramp-up testing.

The orchestrator is one LLM tool-calling loop followed by a mode-specific step:
`run_full_diagnostic` measures a fixed burst, and `find_breaking_point` ramps
offered load until the API is completely dead (100% failures or a refused
connection).

Local mode (the target is loopback):
1. Send the user's goal, plus the `run_full_diagnostic` and
   `find_breaking_point` tool schemas, to Groq.
2. If the model asks for the tool, execute it locally (load test + CPU/memory
   profiling) and feed the JSON result back as a `tool` message.
3. Repeat until the model answers without requesting another tool. That answer
   is the diagnosis.
4. Resolve the file that declares the measured route, read it, and call
   `suggester.suggest_code_fix(diagnosis, source)`, so the fix is written
   against the real framework instead of a guess. The answer is the diagnosis
   plus one concrete code/infrastructure fix, optionally verified on staging.

Remote Audit mode (the target is any non-loopback host):
1. The same tool loop measures the API.
2. The model audits it instead of diagnosing code it cannot see: no source
   lookup, no `suggester.suggest_code_fix`, no patch, no staging.
3. `build_audit_report` adds a deterministic verdict (`Broken`, `Degraded`,
   `Stable`, or `Inconclusive`) so a model that forgets the rule cannot hide a
   broken API. Failed requests and p95 latency above 2 seconds are the audit
   thresholds.

Mode is selected from the first URL in the prompt; if the prompt has no URL,
the URL the diagnostic actually measured decides, and the run switches to Audit
mode before the final answer.

Groq exposes an OpenAI-compatible API, so the official `openai` SDK is used
with `base_url` pointing at `https://api.groq.com/openai/v1`.

Usage:
    py agent.py "Diagnose http://127.0.0.1:8000/heavy with 12 requests"
    py agent.py --source path/to/app.py "Diagnose http://host:8000/heavy"
    py agent.py --verify "Diagnose http://127.0.0.1:8000/heavy"   # skip the prompt
    py agent.py "Audit https://httpbin.org/status/500 with 5 requests"
    py agent.py "Find the breaking point for http://127.0.0.1:8000/slow from 5 RPS and push until it crashes"
    py agent.py            # runs the built-in demo prompt
"""

import ipaddress
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI

from feedback_loop import format_comparison, run_feedback_loop
from llm_client import MODEL, create_client
from suggester import SUGGESTION_HEADING, suggest_code_fix
from tools import find_breaking_point, run_full_diagnostic

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
    "If the user asks to find the breaking point or test the capacity limits, "
    "use the find_breaking_point tool instead of run_full_diagnostic. You are a "
    "ruthless performance tester: do not stop at the first sign of trouble. "
    "Mild latency and a few failed requests are not reasons to stop; keep "
    "hammering the API until it is completely dead. The tool only ends the ramp "
    "when an entire step fails 100% of its requests or the target refuses the "
    "connection. Report the exact RPS where it died and the reason, for example "
    "'The API completely crashed at 160 RPS with connection refused'. If "
    "crash_reason is null, say the API survived max_rps and report the last "
    "successful rate, and note that max_rps is this Python client's practical "
    "ceiling (about 1,000 req/s) rather than proof the server has more room. If "
    "break_was_client_limited is true, say the 10-worker load generator was "
    "queueing, so the crash rate is a lower bound on what the target could be "
    "driven to, not a hardware capacity figure. "
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

# Two operating modes. Local mode may read source and propose a fix because the
# target runs on this machine. Remote targets expose no source and must never
# receive a patch the agent cannot verify, so they are audited instead.
MODE_LOCAL = "local"
MODE_AUDIT = "audit"

AUDIT_VERDICT_BROKEN = "Broken"
AUDIT_VERDICT_DEGRADED = "Degraded"
AUDIT_VERDICT_STABLE = "Stable"
AUDIT_VERDICT_UNKNOWN = "Inconclusive"

# Audit thresholds are policy, not a measured SLO: any failed request means the
# API is not serving reliably, and a p95 above two seconds means it answers too
# slowly under load. Keeping them in code (not only in the prompt) makes the
# verdict reproducible even when the model is vague.
AUDIT_P95_THRESHOLD_SECONDS = 2.0

# System prompt for targets that are not on this machine. It deliberately has no
# source-code or remediation stage: an audit describes health, it does not patch
# a service the agent cannot see.
AUDIT_SYSTEM_PROMPT = (
    "You are PerfAgent in REMOTE AUDIT mode. The target API does not run on this "
    "machine, so you are a read-only auditor: measure its health and report what "
    "the numbers prove. You cannot see, read, or change its source code.\n"
    "Whenever the user asks you to measure, test, benchmark, or audit an "
    "endpoint, you must call the run_full_diagnostic tool instead of estimating "
    "results yourself. The tool load-tests the endpoint while sampling CPU and "
    "memory and returns latency, success/failure counts, throughput, and an idle "
    "baseline. Never invent latency, throughput, CPU, memory, or failure "
    "numbers. If the target URL is missing, ask the user for it.\n"
    "If the user asks for the breaking point or capacity limits, call "
    "find_breaking_point instead. You are a ruthless performance tester: do not "
    "stop at the first sign of trouble. Keep the load climbing until the API is "
    "completely dead; the tool stops only when a step fails 100% of its requests "
    "or the connection is refused. Report the exact RPS where it died and the "
    "reason, for example 'The API completely crashed at 160 RPS with connection "
    "refused'. If crash_reason is null, say the API survived max_rps and report "
    "the last successful rate; max_rps is the Python client's practical ceiling "
    "(about 1,000 req/s), not proof of server headroom. If "
    "break_was_client_limited is true, say the 10-worker generator was the "
    "queue, so the crash rate is a lower bound. Report the API's own "
    "avg_latency_seconds separately. Do not suggest a fix for a capacity limit "
    "either.\n"
    "After the tool returns, classify the API and state the verdict explicitly:\n"
    "- Broken: any request failed (load_test.failed > 0). A timeout, a "
    "connection error, or any non-200 status is a failure, so the API is not "
    "serving reliably even when most requests succeeded.\n"
    "- Degraded: no request failed, but load_test.p95_latency_seconds is above "
    "2.0 seconds. The API answers, but too slowly under load.\n"
    "- Stable: no request failed and p95_latency_seconds is at or below 2.0 "
    "seconds.\n"
    "Begin the answer with one line in exactly this form: '**Verdict:** Broken', "
    "'**Verdict:** Degraded', or '**Verdict:** Stable'. Then report the failure "
    "rate (failed / total attempts), successful and failed counts, average, p95 "
    "and max latency, throughput, and the CPU and memory deltas. Name the "
    "weakest points the measurement supports: error responses, timeouts, "
    "saturation, unstable latency, or a slow downstream dependency. Rank them by "
    "measured impact and keep the answer concise and technical.\n"
    "Do NOT write code, a diff, a patch, a remediation plan, or a suggested fix. "
    "Do NOT mention or look for local source files. Remote code is out of scope: "
    "your job ends with the audit report.\n"
    "If the diagnostic returns an error instead of load numbers, say the audit "
    "is Inconclusive and explain what failed; never guess a verdict."
)

# Appended when the prompt had no URL and the tool measured a remote target, so
# the model's final answer follows the audit rules even though the first system
# prompt was the local one.
AUDIT_MODE_NOTICE = (
    "MODE CHANGE: the URL just measured is remote, so this run is now in REMOTE "
    "AUDIT mode. Classify the API as Broken (any failed request), Degraded (no "
    "failures but p95 latency above 2.0 seconds), or Stable (no failures and p95 "
    "at or below 2.0 seconds), and report the verdict and failure rate. Do NOT "
    "write code, a diff, a patch, or a suggested fix, and do NOT look for local "
    "source files."
)

# Any heading the model might use if it ignores the audit instruction; the report
# builder cuts the answer at the first match so a code fix cannot leak through.
_SUGGESTION_SECTION = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*|\*\*)?Suggested[ \t]+Fix\b",
    re.IGNORECASE | re.MULTILINE,
)

# A verdict line is regenerated from the measurements, so model-authored ones are
# removed rather than trusted (a model can otherwise call a 100% failure "Stable").
_MODEL_VERDICT_LINE = re.compile(
    r"^[ \t]*\*\*Verdict:\*\*.*(?:\n|$)", re.IGNORECASE | re.MULTILINE
)

_URL_IN_TEXT = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)


def _first_url(text: str) -> str | None:
    """Return the first absolute URL in `text`, or None when there is none."""
    match = _URL_IN_TEXT.search(text or "")
    return match.group(0) if match else None


def is_local_url(url: str | None) -> bool:
    """True only for a loopback target that this agent could inspect locally.

    Loopback is the narrow, defensible definition of "local": it is the one host
    guaranteed to be this machine, so reading its source is meaningful. A private
    LAN address is treated as remote because the code may live on another host.
    """
    if not isinstance(url, str) or not url.strip():
        return False
    host = urlparse(url).hostname
    if not host:
        return False
    host = host.lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def detect_mode(user_prompt: str) -> str:
    """Pick the operating mode from the first URL in the prompt.

    With no URL the local prompt is used, because the agent has nothing to judge
    yet. If the eventual measurement is remote, `run_agent` switches to Audit
    mode before the final answer.
    """
    url = _first_url(user_prompt)
    if url is None:
        return MODE_LOCAL
    return MODE_LOCAL if is_local_url(url) else MODE_AUDIT


def _number(source: dict[str, Any], key: str) -> float:
    """Read a numeric metric without letting a malformed payload crash a report."""
    value = source.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def classify_api_health(diagnostic_result: dict[str, Any]) -> tuple[str, str]:
    """Return `(verdict, reason)` from a diagnostic payload.

    This is the audit's source of truth: the verdict is derived from measured
    failures and p95 latency, not from model judgement. `Inconclusive` is used
    when no load numbers exist, because inventing a health verdict from an error
    would be worse than admitting the audit could not run.
    """
    load_test = diagnostic_result.get("load_test")
    if not isinstance(load_test, dict) or not load_test:
        error = diagnostic_result.get("error") or "no load-test payload"
        return (
            AUDIT_VERDICT_UNKNOWN,
            f"the diagnostic produced no measurements ({error})",
        )

    failed = int(_number(load_test, "failed"))
    successful = int(_number(load_test, "successful"))
    total = failed + successful
    if total <= 0:
        return AUDIT_VERDICT_UNKNOWN, "the diagnostic recorded no request attempts"

    failure_rate = failed / total * 100
    p95 = _number(load_test, "p95_latency_seconds")

    if failed > 0:
        return (
            AUDIT_VERDICT_BROKEN,
            f"{failed} of {total} requests failed ({failure_rate:.1f}% failure rate)",
        )
    if p95 > AUDIT_P95_THRESHOLD_SECONDS:
        return (
            AUDIT_VERDICT_DEGRADED,
            f"no requests failed, but p95 latency was {p95:.2f} s "
            f"(above the {AUDIT_P95_THRESHOLD_SECONDS:.2f} s threshold)",
        )
    return (
        AUDIT_VERDICT_STABLE,
        f"no requests failed and p95 latency was {p95:.2f} s "
        f"(within the {AUDIT_P95_THRESHOLD_SECONDS:.2f} s threshold)",
    )


def _clean_audit_analysis(analysis: str) -> str:
    """Remove model-authored verdicts and any suggested-fix section.

    The prompt asks for both rules, but a strict audit must not depend on model
    compliance: the computed verdict replaces the model's, and a trailing fix is
    cut before it reaches the user.
    """
    text = _MODEL_VERDICT_LINE.sub("", analysis or "")
    match = _SUGGESTION_SECTION.search(text)
    if match:
        text = text[: match.start()]
    return text.strip()


def build_audit_report(analysis: str, diagnostic_result: dict[str, Any]) -> str:
    """Compose the Remote Audit Report from measurements plus model analysis.

    The verdict, failure rate, and latency figures are generated here so the
    report is complete even if the model's prose is vague or wrong. The model's
    analysis remains the qualitative section: weak points and their ranking.
    """
    verdict, reason = classify_api_health(diagnostic_result)
    load_test = diagnostic_result.get("load_test")
    lines = ["## Audit Report", "", f"**Verdict:** {verdict}"]

    if isinstance(load_test, dict) and load_test:
        failed = int(_number(load_test, "failed"))
        successful = int(_number(load_test, "successful"))
        total = failed + successful
        failure_rate = failed / total * 100 if total else 0.0
        avg = _number(load_test, "avg_latency_seconds")
        p95 = _number(load_test, "p95_latency_seconds")
        maximum = _number(load_test, "max_latency_seconds")
        throughput = _number(load_test, "requests_per_second")
        target = diagnostic_result.get("url") or load_test.get("url")
        cpu_delta = _number(diagnostic_result, "cpu_delta_percent")

        lines.append("")
        if target:
            lines.append(f"- **Target:** {target}")
        lines.extend(
            [
                f"- **Failure rate:** {failure_rate:.1f}% "
                f"({failed} failed of {total})",
                f"- **Latency:** avg {avg * 1000:.0f} ms, "
                f"p95 {p95 * 1000:.0f} ms, max {maximum * 1000:.0f} ms",
                f"- **Throughput:** {throughput:.2f} req/s",
                # On a remote API these readings describe the load generator's
                # host, not the server, so they are context rather than evidence
                # about the target.
                f"- **Local CPU delta:** {cpu_delta:+.2f} points "
                "(load-generator host, not the remote server)",
            ]
        )

    lines.extend(["", f"**Why:** {reason}", ""])
    cleaned = _clean_audit_analysis(analysis)
    if cleaned:
        lines.extend(["### Analysis", "", cleaned])
    return "\n".join(lines).strip()


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
    },
    {
        "type": "function",
        "function": {
            "name": "find_breaking_point",
            "description": (
                "Find the request rate at which an API is completely dead. Ramps "
                "offered load from start_rps to max_rps, doubling each step by "
                "default (10 -> 20 -> 40 -> 80) or adding step_increase when it "
                "is given, with a one-second burst per rate. Mild latency and "
                "partial failures are ignored: the ramp only stops when 100% of "
                "a step's requests fail or the target refuses the connection. "
                "Returns breaking_rps, last_successful_rps, crash_reason (for "
                "example '100% HTTP 500s' or 'Connection Refused'), the final "
                "metrics, every attempted step, and whether the 10-worker load "
                "generator itself was queueing (client_limited). max_rps "
                "defaults to 1000, this Python client's practical ceiling; "
                "20,000+ RPS needs a dedicated generator such as k6."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "Absolute URL to ramp, including scheme and port "
                            "(for example http://127.0.0.1:8000/slow)."
                        ),
                    },
                    "start_rps": {
                        "type": "integer",
                        "description": "First offered request rate. Minimum 1.",
                        "minimum": 1,
                    },
                    "step_increase": {
                        "type": "integer",
                        "description": (
                            "Optional RPS added after each surviving step. Omit "
                            "it to double the rate each step (10 -> 20 -> 40 -> "
                            "80), which reaches the crash point fastest. Minimum "
                            "1 when provided."
                        ),
                        "minimum": 1,
                    },
                    "max_rps": {
                        "type": "integer",
                        "description": (
                            "Highest offered rate to try before giving up. "
                            "Defaults to 1000, the practical ceiling of this "
                            "Python client; 20,000+ RPS requires k6 or similar. "
                            "Must be at least start_rps."
                        ),
                        "minimum": 1,
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
]


def _parse_arguments(raw_arguments: str) -> dict[str, Any]:
    """Decode the model's JSON argument string, tolerating malformed output."""
    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _integer_argument(
    arguments: dict[str, Any], key: str, default: int
) -> int:
    """Read an integer tool argument, rejecting booleans and non-numeric text."""
    value = arguments.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run a tool by name and return a JSON-serialisable result.

    Tool failures are returned as data rather than raised so the model can
    correct its arguments or report the problem instead of crashing the run.
    """
    if name not in {"run_full_diagnostic", "find_breaking_point"}:
        return {"error": f"Unknown tool: {name}"}

    url = arguments.get("url")
    if not isinstance(url, str) or not url:
        return {"error": f"{name} requires a non-empty 'url' argument"}

    if name == "find_breaking_point":
        try:
            raw_step = arguments.get("step_increase")
            return find_breaking_point(
                url=url,
                start_rps=_integer_argument(arguments, "start_rps", 10),
                # Omitted step_increase means "double each step"; an explicit
                # value keeps the finer additive table available.
                step_increase=(
                    None
                    if raw_step is None
                    else _integer_argument(arguments, "step_increase", 10)
                ),
                max_rps=_integer_argument(arguments, "max_rps", 1000),
            )
        except (TypeError, ValueError) as exc:
            return {"error": f"Invalid arguments for find_breaking_point: {exc}"}

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


def source_file_for(
    url: str | None, source_file: str | Path | None = None
) -> Path | None:
    """Resolve the file once, so reading it and staging it agree.

    Phase 5 needs the same file the suggester was shown: resolving twice could
    textually disagree if the project changes between calls.
    """
    return _resolve_source_file(source_file, url)


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
    source_context: str,
) -> tuple[str, str | None]:
    """Return `diagnosis` plus one concrete remediation, and that fix on its own.

    The suggestion is a second, narrowly scoped model call that receives the
    target's source code, which is what keeps the fix on the same framework the
    service is actually built on. A provider failure here must not discard a
    completed measurement, so the error is reported in the answer itself instead
    of being raised - and the fix is `None`, which is also what tells the Phase 5
    feedback loop there is nothing to stage.
    """
    print("🩺 Asking the model for a concrete fix...")
    try:
        fix = suggest_code_fix(diagnosis, source_context, client=client)
    except Exception as exc:  # noqa: BLE001 - keep the diagnosis, surface the cause
        print(f"⚠️  Suggestion step failed: {exc}")
        return (
            f"**Diagnosis**\n\n{diagnosis}\n\n{SUGGESTION_HEADING}\n\n"
            f"_Suggestion unavailable: {exc}_",
            None,
        )

    report = f"**Diagnosis**\n\n{diagnosis}\n\n{SUGGESTION_HEADING}\n\n{fix}"
    return report, fix


def _ask_to_verify() -> bool:
    """Ask whether to verify on staging, without ever blocking a non-terminal run."""
    if sys.stdin is None or not sys.stdin.isatty():
        print(
            "↪️  Staging verification skipped: no interactive terminal. "
            "Re-run with --verify to apply the fix to staging automatically."
        )
        return False
    try:
        answer = input(
            "Would you like to apply this fix to staging and verify it? (y/n) "
        )
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer.strip().lower() in {"y", "yes"}


def _maybe_verify(
    *,
    report: str,
    fix: str | None,
    diagnostic_result: dict[str, Any],
    source_path: Path | None,
    source_context: str,
    client: OpenAI,
    verify: bool | None,
) -> str:
    """Optionally prove the fix on a staging copy and append the before/after.

    The measurement is the point: a fix that is never re-tested is a hypothesis.
    Every failure path here returns the diagnosis plus an explanation instead of
    raising, because losing a completed diagnosis to a staging problem would be
    the worse outcome - and the original file is never written, so declining or
    failing is always safe.
    """
    if not fix or not diagnostic_result.get("load_test"):
        # Nothing was measured, or nothing was suggested: nothing to prove.
        return report
    if verify is False:
        return report
    if source_path is None:
        print(
            "🧪 Staging verification skipped: the measured file could not be "
            "resolved, so there is nothing safe to copy."
        )
        return report
    if verify is not True and not _ask_to_verify():
        return report

    print(f"\n🧪 Verifying the fix on a staging copy of {source_path.name}...")
    try:
        result = run_feedback_loop(
            diagnostic_result.get("url") or "",
            source_path,
            fix,
            diagnostic_result,
            source_context,
            client=client,
        )
    except Exception as exc:  # noqa: BLE001 - keep the diagnosis, report the cause
        print(f"❌ Staging verification failed: {exc}")
        section = (
            "## Staging verification\n\n"
            f"Verification could not run: {exc}\n\n"
            "The diagnosis and suggested fix above are unaffected, and "
            f"`{source_path.name}` was not modified."
        )
    else:
        section = format_comparison(
            result["comparison"],
            url=result["url"],
            original_file=Path(result["original_file"]).name,
            staged_file=Path(result["staging_file"]).name,
            total_requests=result["total_requests"],
        )
        if result.get("restore_error"):
            section += f"\n\n⚠️ Cleanup problem: {result['restore_error']}"
    print(section)
    return f"{report}\n\n{section}"


def run_agent(
    user_prompt: str,
    client: OpenAI | None = None,
    source_file: str | Path | None = None,
    verify: bool | None = False,
    mode: str | None = None,
) -> str:
    """Measure `user_prompt`'s target, then fix it locally or audit it remotely.

    Local mode (loopback target) yields a diagnosis, then resolves and reads the
    measured route's file so `suggester.suggest_code_fix` can propose a concrete
    fix against the real framework. Phase 5 optionally verifies that fix on a
    staging copy: `verify=True` applies it, `verify=False` (the default for
    callers) skips it, and `None` asks first - which the CLI does only in a real
    terminal, so a library caller or a piped run never blocks on `input()`.

    Remote Audit mode (any non-loopback target) skips all of that. The final
    answer is `build_audit_report`, whose verdict is computed from the measured
    failures and p95 latency. No source lookup, no suggester call, and no staging
    run: there is no local code to read and no safe way to apply a fix.

    `mode` forces `MODE_LOCAL` or `MODE_AUDIT`; by default the first URL in the
    prompt decides, and a remote URL measured later still switches the run to
    Audit mode before the final answer.
    """
    if mode is not None and mode not in {MODE_LOCAL, MODE_AUDIT}:
        raise ValueError(f"unknown mode: {mode!r}")

    client = client or create_client()
    audit_mode = (
        mode == MODE_AUDIT
        if mode is not None
        else detect_mode(user_prompt) == MODE_AUDIT
    )
    print(f"👤 User: {user_prompt}\n")
    if audit_mode:
        print(
            "🔒 Remote Audit mode: measuring only; no source lookup, no code fix, "
            "no staging verification."
        )
        if source_file:
            print("↪️  --source is ignored in Audit mode (the target is remote).")
        if verify:
            print("↪️  --verify is ignored in Audit mode (nothing is applied).")

    messages: list[Any] = [
        {
            "role": "system",
            "content": AUDIT_SYSTEM_PROMPT if audit_mode else SYSTEM_PROMPT,
        },
        {"role": "user", "content": user_prompt},
    ]
    diagnostic_ran = False
    diagnostic_url: str | None = None
    diagnostic_result: dict[str, Any] = {}
    ramp_ran = False
    ramp_result: dict[str, Any] = {}
    mode_switched = False

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
            if ramp_ran:
                # A capacity ramp is its own report. Sending it through the
                # audit builder would look for `load_test` and wrongly call the
                # run Inconclusive, while the local path would ask for a code fix
                # when the user only asked where capacity ends.
                print(
                    "🤖 PerfAgent (breaking point analysis, measured "
                    f"breaking_rps={ramp_result.get('breaking_rps')}):\n"
                    f"{analysis}"
                )
                return analysis
            if audit_mode:
                # Strict remote audit: the report is built from the measurements
                # and the model's prose. No source lookup, no suggester, no patch.
                # This runs even when the diagnostic failed, so an unreachable
                # API is reported as Inconclusive rather than as a diagnosis.
                report = build_audit_report(analysis, diagnostic_result)
                print(f"🤖 PerfAgent (audit report):\n{report}")
                return report

            if not diagnostic_ran:
                # Nothing was measured, so there is no bottleneck to remediate.
                print(f"🤖 PerfAgent: {analysis}")
                return analysis

            # Resolve once: the suggester and the staging loop must be looking at
            # the same file, or the "verified" copy is not the one that was fixed.
            source_path = source_file_for(diagnostic_url, source_file)
            source_context = load_source_context(None, source_path)
            report, fix = _with_suggested_fix(analysis, client, source_context)
            print(f"🤖 PerfAgent (diagnosis + suggested fix):\n{report}")
            return _maybe_verify(
                report=report,
                fix=fix,
                diagnostic_result=diagnostic_result,
                source_path=source_path,
                source_context=source_context,
                client=client,
                verify=verify,
            )

        # The assistant turn must be echoed back before its tool results,
        # otherwise the next request is rejected for an orphaned tool_call_id.
        messages.append(message)

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = _parse_arguments(tool_call.function.arguments)
            print(f"🛠️  Calling {name} with {arguments}")

            result = _execute_tool(name, arguments)
            print(f"   ↳ Result: {json.dumps(result)}")
            if name == "find_breaking_point":
                ramp_ran = True
                ramp_result = result
            # Only a diagnostic that actually produced load numbers justifies a
            # remediation; a failed run has no measured bottleneck to explain.
            # The failed payload is still kept so Audit mode can explain it.
            if name == "run_full_diagnostic":
                diagnostic_url = result.get("url") or arguments.get("url")
                diagnostic_result = result
                diagnostic_ran = bool(result.get("load_test"))
                # The prompt may not have contained a URL. The measured target
                # still decides the mode, so a remote run cannot fall through to
                # the local source/suggester path.
                if not audit_mode and diagnostic_url and not is_local_url(
                    diagnostic_url
                ):
                    audit_mode = True
                    mode_switched = True

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": json.dumps(result),
                }
            )

        if mode_switched:
            messages.append({"role": "system", "content": AUDIT_MODE_NOTICE})
            print(
                "🔒 Measured URL is remote; switching to Remote Audit mode "
                "(no source lookup, no code fix)."
            )
            mode_switched = False

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

    # `--verify` answers the staging prompt in advance (unattended runs);
    # `--no-verify` declines it. Without either flag, the prompt appears only in
    # an interactive terminal.
    verify: bool | None = None
    if "--verify" in argv:
        argv.remove("--verify")
        verify = True
    if "--no-verify" in argv:
        argv.remove("--no-verify")
        verify = False

    prompt = " ".join(argv).strip() or DEFAULT_PROMPT
    # The target API must be running, e.g.:
    #   py -m uvicorn dummy_api:app --reload
    try:
        run_agent(prompt, source_file=source_file, verify=verify)
    except Exception as exc:  # CLI boundary: report the failure, exit non-zero
        print(f"❌ {exc}")
        raise SystemExit(1) from exc


