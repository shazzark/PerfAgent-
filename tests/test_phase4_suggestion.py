"""Phase 4 orchestration tests: diagnosis -> automatic code suggestion.

The fake client scripts the two model calls `run_agent` makes (the tool-calling
loop and the remediation call), so these tests exercise the real wiring without
network access, API cost, or dependence on model behaviour or host load.

Phase 4.5 adds the second half of the contract: the remediation call is never
made blindfolded. These tests pin the source-context resolution (measured route
-> declaring file, explicit override, env var, graceful degradation) and the
prompt that carries the code, because a suggester that cannot see the framework
invents one - the live Phase 4 run produced a Flask diff for a FastAPI service.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent
import suggester

DIAGNOSTIC_RESULT = {
    "url": "http://127.0.0.1:8000/heavy",
    "total_requests": 12,
    "load_test": {
        "total_time_seconds": 3.12,
        "successful": 12,
        "failed": 0,
        "avg_latency_seconds": 1.591,
        "p95_latency_seconds": 1.613,
        "max_latency_seconds": 1.613,
    },
    "cpu_delta_percent": 23.7,
}

DIAGNOSIS = (
    "CPU rose from 24% idle to a 96% peak while the 12 requests were in flight "
    "and memory stayed flat, so /heavy is CPU-bound."
)

FIX = "Offload the hashing loop.\n\n```diff\n-v = compute()\n+v = await loop.run_in_executor(pool, compute)\n```"

# What the orchestrator attaches in production: a filename header, then the file.
SOURCE_CONTEXT = (
    "# File: app.py\n"
    "from fastapi import FastAPI\n"
    "\n"
    "app = FastAPI()\n"
    "\n"
    '@app.get("/heavy")\n'
    "def heavy():\n"
    "    return {\"iterations\": _burn_cpu(1.5)}\n"
)


def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _FakeCompletions:
    def __init__(self, script):
        self._script = script
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._script(kwargs)


class FakeClient:
    """Minimal stand-in for `openai.OpenAI` with a scripted `completions.create`."""

    def __init__(self, script):
        self.chat = SimpleNamespace(completions=_FakeCompletions(script))

    @property
    def calls(self):
        return self.chat.completions.calls


def _is_suggestion_call(kwargs):
    """True when the request is the remediation call, not the tool loop."""
    return kwargs["messages"][0]["content"] == suggester.SUGGEST_SYSTEM_PROMPT


def _role(message):
    """Read a role from either a chat message object or a plain dict."""
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _diagnose_then_answer(analysis=DIAGNOSIS):
    """Script: request the diagnostic tool once, then answer with `analysis`."""

    def script(kwargs):
        if _is_suggestion_call(kwargs):
            return _response(_message(content=FIX))

        saw_tool_result = any(_role(message) == "tool" for message in kwargs["messages"])
        if saw_tool_result:
            return _response(_message(content=analysis))
        return _response(
            _message(
                tool_calls=[
                    _tool_call(
                        "run_full_diagnostic",
                        json.dumps(
                            {
                                "url": "http://127.0.0.1:8000/heavy",
                                "total_requests": 12,
                            }
                        ),
                    )
                ]
            )
        )

    return script


def _run_agent(prompt, client, **kwargs):
    """Run the orchestrator with its console trace suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return agent.run_agent(prompt, client=client, **kwargs)


class RunAgentSuggestionTests(unittest.TestCase):
    def test_suggestion_is_appended_to_the_diagnosis(self):
        client = FakeClient(_diagnose_then_answer())

        with patch("agent.run_full_diagnostic", return_value=DIAGNOSTIC_RESULT):
            report = _run_agent("diagnose /heavy", client)

        self.assertIn(DIAGNOSIS, report)
        self.assertIn("**Diagnosis**", report)
        self.assertIn(suggester.SUGGESTION_HEADING, report)
        self.assertIn(FIX, report)
        # The composed answer must put the diagnosis before the fix.
        self.assertLess(report.index(DIAGNOSIS), report.index(FIX))

    def test_suggestion_receives_the_exact_diagnosis(self):
        client = FakeClient(_diagnose_then_answer())

        with patch.dict("os.environ", {agent.SOURCE_FILE_ENV_VAR: ""}), patch(
            "agent.run_full_diagnostic", return_value=DIAGNOSTIC_RESULT
        ):
            _run_agent("diagnose /heavy", client)

        suggestion_prompts = [
            call["messages"][1]["content"]
            for call in client.calls
            if _is_suggestion_call(call)
        ]
        self.assertEqual(len(suggestion_prompts), 1)
        self.assertIn(DIAGNOSIS, suggestion_prompts[0])
        self.assertTrue(suggestion_prompts[0].startswith("Based on this bottleneck:"))
        self.assertIn("markdown code block", suggestion_prompts[0])
        # Phase 4.5: the measured /heavy route resolves to the FastAPI target, so
        # the prompt carries that file rather than only the prose diagnosis.
        self.assertIn("# File: dummy_api.py", suggestion_prompts[0])
        self.assertIn("from fastapi import FastAPI", suggestion_prompts[0])

    def test_conversational_answer_is_not_sent_for_a_fix(self):
        client = FakeClient(
            lambda kwargs: _response(_message(content="I can test APIs."))
        )

        report = _run_agent("what do you do?", client)

        self.assertEqual(report, "I can test APIs.")
        self.assertEqual(len(client.calls), 1)

    def test_failed_diagnostic_does_not_trigger_a_suggestion(self):
        failed = {
            "url": "http://127.0.0.1:8000/heavy",
            "load_test": None,
            "error": "boom",
        }
        base_script = _diagnose_then_answer()

        def script(kwargs):
            if _is_suggestion_call(kwargs):
                return _response(_message(content=FIX))
            return base_script(kwargs)

        client = FakeClient(script)
        with patch("agent.run_full_diagnostic", return_value=failed):
            report = _run_agent("diagnose /heavy", client)

        self.assertEqual(report, DIAGNOSIS)
        self.assertFalse(any(_is_suggestion_call(call) for call in client.calls))

    def test_suggestion_failure_keeps_the_diagnosis(self):
        base_script = _diagnose_then_answer()

        def script(kwargs):
            if _is_suggestion_call(kwargs):
                raise RuntimeError("provider unavailable")
            return base_script(kwargs)

        client = FakeClient(script)
        with patch("agent.run_full_diagnostic", return_value=DIAGNOSTIC_RESULT):
            report = _run_agent("diagnose /heavy", client)

        self.assertIn(DIAGNOSIS, report)
        self.assertIn("Suggestion unavailable: provider unavailable", report)


class SourceContextTests(unittest.TestCase):
    """Phase 4.5: the remediation call always gets the target's code.

    `agent.suggest_code_fix` is patched out here because this class is about the
    resolution and reading that happens *before* the model call; the prompt that
    carries the result is asserted in `SuggestCodeFixTests`.
    """

    def setUp(self):
        # An empty value means "no override", so these tests do not depend on the
        # developer's environment.
        env = patch.dict("os.environ", {agent.SOURCE_FILE_ENV_VAR: ""})
        env.start()
        self.addCleanup(env.stop)

    def _captured_source_context(self, **kwargs):
        client = FakeClient(_diagnose_then_answer())

        with patch("agent.run_full_diagnostic", return_value=DIAGNOSTIC_RESULT), patch(
            "agent.suggest_code_fix", return_value=FIX
        ) as suggest:
            report = _run_agent("diagnose /heavy", client, **kwargs)

        self.assertIn(FIX, report)
        self.assertEqual(suggest.call_count, 1)
        diagnosis, source_context = suggest.call_args.args
        self.assertEqual(diagnosis, DIAGNOSIS)
        return source_context

    def test_measured_endpoint_resolves_to_its_fastapi_file(self):
        context = self._captured_source_context()

        self.assertTrue(context.startswith("# File: dummy_api.py"))
        self.assertIn("from fastapi import FastAPI", context)
        self.assertIn('@app.get("/heavy")', context)

    def test_explicit_source_file_overrides_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "my_api.py"
            path.write_text(
                'from fastapi import APIRouter\n\nrouter = APIRouter()\n\n'
                '@router.post("/jobs")\nasync def jobs():\n    ...\n',
                encoding="utf-8",
            )

            context = self._captured_source_context(source_file=path)

        self.assertTrue(context.startswith("# File: my_api.py"))
        self.assertIn('@router.post("/jobs")', context)
        self.assertNotIn("dummy_api", context)

    def test_environment_variable_supplies_the_file(self):
        with patch.dict("os.environ", {agent.SOURCE_FILE_ENV_VAR: "dummy_api.py"}):
            context = self._captured_source_context()

        self.assertIn("from fastapi import FastAPI", context)

    def test_missing_source_file_degrades_to_empty_context(self):
        context = self._captured_source_context(source_file="does_not_exist.py")

        self.assertEqual(context, "")

    def test_unknown_endpoint_degrades_to_empty_context(self):
        self.assertIsNone(agent.find_source_file("http://127.0.0.1:8000/nope"))

    def test_oversized_source_is_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big_api.py"
            path.write_text(
                "from fastapi import FastAPI\n" + "# padding line\n" * 4000,
                encoding="utf-8",
            )

            context = self._captured_source_context(source_file=path)

        self.assertIn("source truncated", context)
        # The cap applies to the file, plus the small filename header/marker.
        self.assertLess(len(context), agent.MAX_SOURCE_CONTEXT_CHARS + 200)


class SuggestCodeFixTests(unittest.TestCase):
    def test_prompt_contains_the_diagnosis_the_code_and_the_output_contract(self):
        captured = {}

        def script(kwargs):
            captured.update(kwargs)
            return _response(_message(content=FIX))

        fix = suggester.suggest_code_fix(
            "CPU-bound handler", SOURCE_CONTEXT, client=FakeClient(script)
        )

        self.assertEqual(fix, FIX)
        self.assertEqual(captured["model"], agent.MODEL)
        prompt = captured["messages"][1]["content"]
        self.assertIn("Based on this bottleneck: CPU-bound handler", prompt)
        # The code is fenced verbatim, so the model reads the real framework.
        self.assertIn(f"```\n{SOURCE_CONTEXT.strip()}\n```", prompt)
        self.assertIn("Keep it brief.", prompt)

    def test_source_context_is_required(self):
        # A missing argument is a programming error, not a silent blindfold.
        with self.assertRaises(TypeError):
            suggester.suggest_code_fix("CPU-bound handler")

    def test_non_string_source_context_is_rejected(self):
        with self.assertRaises(ValueError):
            suggester.suggest_code_fix("CPU-bound handler", None)

    def test_blank_source_context_is_marked_unknown(self):
        captured = {}

        def script(kwargs):
            captured.update(kwargs)
            return _response(_message(content=FIX))

        suggester.suggest_code_fix("CPU-bound handler", "   ", client=FakeClient(script))

        self.assertIn(suggester.NO_SOURCE_CONTEXT, captured["messages"][1]["content"])

    def test_empty_diagnosis_is_rejected(self):
        with self.assertRaises(ValueError):
            suggester.suggest_code_fix("   ", SOURCE_CONTEXT)

    def test_empty_model_reply_is_an_error(self):
        client = FakeClient(lambda kwargs: _response(_message(content=None)))

        with self.assertRaises(RuntimeError):
            suggester.suggest_code_fix("CPU-bound handler", SOURCE_CONTEXT, client=client)


if __name__ == "__main__":
    unittest.main()
