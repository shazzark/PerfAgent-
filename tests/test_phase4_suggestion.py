"""Phase 4 orchestration tests: diagnosis -> automatic code suggestion.

The fake client scripts the two model calls `run_agent` makes (the tool-calling
loop and the remediation call), so these tests exercise the real wiring without
network access, API cost, or dependence on model behaviour or host load.
"""

import contextlib
import io
import json
import unittest
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


def _run_agent(prompt, client):
    """Run the orchestrator with its console trace suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return agent.run_agent(prompt, client=client)


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

        with patch("agent.run_full_diagnostic", return_value=DIAGNOSTIC_RESULT):
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


class SuggestCodeFixTests(unittest.TestCase):
    def test_prompt_contains_the_diagnosis_and_the_output_contract(self):
        captured = {}

        def script(kwargs):
            captured.update(kwargs)
            return _response(_message(content=FIX))

        fix = suggester.suggest_code_fix("CPU-bound handler", client=FakeClient(script))

        self.assertEqual(fix, FIX)
        self.assertEqual(captured["model"], agent.MODEL)
        self.assertIn(
            "Based on this bottleneck: CPU-bound handler",
            captured["messages"][1]["content"],
        )
        self.assertIn("Keep it brief.", captured["messages"][1]["content"])

    def test_empty_diagnosis_is_rejected(self):
        with self.assertRaises(ValueError):
            suggester.suggest_code_fix("   ")

    def test_empty_model_reply_is_an_error(self):
        client = FakeClient(lambda kwargs: _response(_message(content=None)))

        with self.assertRaises(RuntimeError):
            suggester.suggest_code_fix("CPU-bound handler", client=client)


if __name__ == "__main__":
    unittest.main()
