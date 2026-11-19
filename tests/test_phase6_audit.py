"""Phase 6 tests: Remote Audit mode for live APIs.

The contract under test is deliberately narrow and structural:

* a non-loopback URL selects the audit system prompt and, when the prompt had no
  URL, the measured URL still switches the run to Audit mode;
* a diagnostic with failures is reported as ``Broken`` even when latency is
  fast, because a returned HTTP 500 is a failure, not a slow success;
* the audit path never resolves source files, calls ``suggester``, or starts a
  staging verification - there is no local code to fix;
* a model-authored verdict or ``Suggested fix`` section cannot override or leak
  into the computed report.

The model and the diagnostic are scripted, so the tests need no API key, no
network, and no running target. Live behaviour against httpbin.org is exercised
separately from the CLI.
"""

import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent

REMOTE_URL = "https://httpbin.org/status/500"
LOCAL_URL = "http://127.0.0.1:8000/heavy"

# 75% failure rate, fast p95: the verdict must key on failures, not latency.
FAILING_RESULT = {
    "url": REMOTE_URL,
    "total_requests": 12,
    "load_test": {
        "url": REMOTE_URL,
        "total_requests": 12,
        "total_time_seconds": 0.8,
        "successful": 3,
        "failed": 9,
        "requests_per_second": 15.0,
        "avg_latency_seconds": 0.31,
        "p95_latency_seconds": 0.72,
        "max_latency_seconds": 0.91,
    },
    "cpu_delta_percent": 1.4,
    "memory_delta_percent": 0.0,
}

LOCAL_RESULT = {
    "url": LOCAL_URL,
    "total_requests": 12,
    "load_test": {
        "url": LOCAL_URL,
        "total_requests": 12,
        "total_time_seconds": 3.1,
        "successful": 12,
        "failed": 0,
        "requests_per_second": 3.8,
        "avg_latency_seconds": 1.59,
        "p95_latency_seconds": 1.61,
        "max_latency_seconds": 1.61,
    },
    "cpu_delta_percent": 23.7,
    "memory_delta_percent": -0.3,
}

AUDIT_ANALYSIS = (
    "Every request returned HTTP 500. Latency is low because the server rejects "
    "the request immediately; the failure rate is the weak point, not speed."
)

LOCAL_ANALYSIS = (
    "CPU rose from a 24% idle baseline to a 96% peak while all 12 requests "
    "succeeded, so /heavy is CPU-bound."
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


def _role(message):
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _content(message):
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


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


def _script(url, analysis):
    """Request the diagnostic once, then answer with `analysis`."""

    def script(kwargs):
        saw_tool_result = any(
            _role(message) == "tool" for message in kwargs["messages"]
        )
        if saw_tool_result:
            return _response(_message(content=analysis))
        return _response(
            _message(
                tool_calls=[
                    _tool_call(
                        "run_full_diagnostic",
                        json.dumps({"url": url, "total_requests": 12}),
                    )
                ]
            )
        )

    return script


def _run_agent(prompt, client, **kwargs):
    """Run the orchestrator with its console trace suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return agent.run_agent(prompt, client=client, **kwargs)


def _result(failed, successful, p95, url=REMOTE_URL):
    return {
        "url": url,
        "load_test": {
            "url": url,
            "successful": successful,
            "failed": failed,
            "p95_latency_seconds": p95,
            "avg_latency_seconds": p95,
            "max_latency_seconds": p95,
            "requests_per_second": 1.0,
        },
    }


class ModeDetectionTests(unittest.TestCase):
    def test_loopback_targets_are_local(self):
        for url in (
            "http://localhost:8000/heavy",
            "http://127.0.0.1:8000/heavy",
            "http://127.0.0.5/heavy",
            "http://[::1]:8000/heavy",
            "http://api.localhost/heavy",
        ):
            with self.subTest(url=url):
                self.assertTrue(agent.is_local_url(url))

    def test_non_loopback_targets_are_remote(self):
        for url in (
            REMOTE_URL,
            "https://api.example.com/v1/health",
            "http://10.0.0.5:8000/heavy",  # LAN host: source may not be here
            "http://localhost.evil.com/heavy",  # lookalike, not loopback
            "http://169.254.169.254/latest/meta-data",
        ):
            with self.subTest(url=url):
                self.assertFalse(agent.is_local_url(url))

    def test_prompt_url_selects_the_mode(self):
        self.assertEqual(
            agent.detect_mode(f"Audit {REMOTE_URL} with 12 requests"),
            agent.MODE_AUDIT,
        )
        self.assertEqual(
            agent.detect_mode(f"Diagnose {LOCAL_URL} with 12 requests"),
            agent.MODE_LOCAL,
        )
        # No URL yet: nothing can be classified, so the local prompt is used and
        # the measured URL decides later.
        self.assertEqual(agent.detect_mode("run a diagnostic"), agent.MODE_LOCAL)


class ApiHealthClassificationTests(unittest.TestCase):
    def test_any_failure_is_broken_even_with_fast_latency(self):
        verdict, reason = agent.classify_api_health(_result(1, 99, 0.05))

        self.assertEqual(verdict, agent.AUDIT_VERDICT_BROKEN)
        self.assertIn("1 of 100 requests failed", reason)
        self.assertIn("1.0% failure rate", reason)

    def test_high_failure_rate_is_broken(self):
        verdict, _ = agent.classify_api_health(_result(9, 3, 0.2))

        self.assertEqual(verdict, agent.AUDIT_VERDICT_BROKEN)

    def test_slow_but_successful_is_degraded(self):
        verdict, reason = agent.classify_api_health(_result(0, 12, 2.5))

        self.assertEqual(verdict, agent.AUDIT_VERDICT_DEGRADED)
        self.assertIn("p95 latency was 2.50 s", reason)

    def test_p95_at_the_threshold_is_stable(self):
        verdict, _ = agent.classify_api_health(_result(0, 12, 2.0))

        self.assertEqual(verdict, agent.AUDIT_VERDICT_STABLE)

    def test_clean_fast_run_is_stable(self):
        verdict, reason = agent.classify_api_health(_result(0, 12, 0.8))

        self.assertEqual(verdict, agent.AUDIT_VERDICT_STABLE)
        self.assertIn("no requests failed", reason)

    def test_missing_measurement_is_inconclusive(self):
        verdict, reason = agent.classify_api_health(
            {"url": REMOTE_URL, "load_test": None, "error": "connection refused"}
        )

        self.assertEqual(verdict, agent.AUDIT_VERDICT_UNKNOWN)
        self.assertIn("connection refused", reason)


class RemoteAuditReportingTests(unittest.TestCase):
    def test_high_failure_rate_reports_broken_and_skips_the_suggester(self):
        client = FakeClient(_script(REMOTE_URL, AUDIT_ANALYSIS))

        with patch("agent.run_full_diagnostic", return_value=FAILING_RESULT), patch(
            "agent.suggest_code_fix"
        ) as suggest, patch("agent.load_source_context") as load, patch(
            "agent.source_file_for"
        ) as resolve, patch(
            "agent.run_feedback_loop"
        ) as feedback:
            report = _run_agent(f"Audit {REMOTE_URL} with 12 requests", client)

        # The report states the verdict explicitly and carries the failure rate.
        self.assertIn("## Audit Report", report)
        self.assertIn("**Verdict:** Broken", report)
        self.assertIn("75.0%", report)
        self.assertIn("9 failed of 12", report)
        self.assertIn(AUDIT_ANALYSIS, report)

        # Audit mode is a structural bypass, not a prompt request.
        suggest.assert_not_called()
        load.assert_not_called()
        resolve.assert_not_called()
        feedback.assert_not_called()
        self.assertEqual(len(client.calls), 2)

    def test_remote_url_selects_the_audit_system_prompt(self):
        client = FakeClient(_script(REMOTE_URL, AUDIT_ANALYSIS))

        with patch("agent.run_full_diagnostic", return_value=FAILING_RESULT):
            _run_agent(f"Audit {REMOTE_URL} with 12 requests", client)

        system_prompt = client.calls[0]["messages"][0]["content"]
        self.assertEqual(system_prompt, agent.AUDIT_SYSTEM_PROMPT)
        self.assertNotEqual(system_prompt, agent.SYSTEM_PROMPT)

    def test_audit_system_prompt_defines_thresholds_and_forbids_code(self):
        prompt = agent.AUDIT_SYSTEM_PROMPT

        for verdict in ("Broken", "Degraded", "Stable", "Inconclusive"):
            self.assertIn(verdict, prompt)
        self.assertIn("load_test.failed > 0", prompt)
        self.assertIn("p95_latency_seconds", prompt)
        self.assertIn("2.0 seconds", prompt)
        self.assertIn("**Verdict:**", prompt)
        self.assertIn("Do NOT write code", prompt)
        self.assertIn("Do NOT mention or look for local source files", prompt)

    def test_model_verdict_and_suggested_fix_cannot_override_the_audit(self):
        # A model that ignores both audit rules must not be able to call a 75%
        # failure "Stable" or append a patch.
        misbehaving = (
            "**Verdict:** Stable\n\n"
            "Most requests returned HTTP 500.\n\n"
            "## Suggested Fix\n\n"
            "```python\nreturn {'ok': True}\n```\n"
        )
        client = FakeClient(_script(REMOTE_URL, misbehaving))

        with patch("agent.run_full_diagnostic", return_value=FAILING_RESULT), patch(
            "agent.suggest_code_fix"
        ) as suggest:
            report = _run_agent(f"Audit {REMOTE_URL} with 12 requests", client)

        self.assertIn("**Verdict:** Broken", report)
        self.assertNotIn("**Verdict:** Stable", report)
        self.assertNotIn("Suggested Fix", report)
        self.assertNotIn("return {'ok': True}", report)
        self.assertIn("Most requests returned HTTP 500.", report)
        suggest.assert_not_called()

    def test_measured_remote_url_switches_mode_when_the_prompt_has_none(self):
        client = FakeClient(_script(REMOTE_URL, AUDIT_ANALYSIS))

        with patch("agent.run_full_diagnostic", return_value=FAILING_RESULT), patch(
            "agent.suggest_code_fix"
        ) as suggest, patch("agent.load_source_context") as load:
            report = _run_agent("run a 12 request diagnostic", client)

        self.assertIn("**Verdict:** Broken", report)
        suggest.assert_not_called()
        load.assert_not_called()
        # The second request carries the mode-change notice, so the model sees
        # the audit rules even though the first system prompt was the local one.
        second_request = client.calls[1]["messages"]
        self.assertTrue(
            any(
                _role(message) == "system"
                and _content(message) == agent.AUDIT_MODE_NOTICE
                for message in second_request
            )
        )

    def test_slow_successful_remote_run_reports_degraded(self):
        degraded = _result(0, 12, 2.5)
        client = FakeClient(_script(REMOTE_URL, "All requests succeeded but slowly."))

        with patch("agent.run_full_diagnostic", return_value=degraded):
            report = _run_agent(f"Audit {REMOTE_URL} with 12 requests", client)

        self.assertIn("**Verdict:** Degraded", report)
        self.assertNotIn("**Verdict:** Broken", report)

    def test_failed_diagnostic_reports_inconclusive_without_a_suggester(self):
        failed = {"url": REMOTE_URL, "load_test": None, "error": "connection refused"}

        def script(kwargs):
            if any(_role(message) == "tool" for message in kwargs["messages"]):
                return _response(_message(content="The target could not be reached."))
            return _response(
                _message(
                    tool_calls=[
                        _tool_call(
                            "run_full_diagnostic",
                            json.dumps({"url": REMOTE_URL, "total_requests": 12}),
                        )
                    ]
                )
            )

        client = FakeClient(script)

        with patch("agent.run_full_diagnostic", return_value=failed), patch(
            "agent.suggest_code_fix"
        ) as suggest:
            report = _run_agent(f"Audit {REMOTE_URL} with 12 requests", client)

        self.assertIn("**Verdict:** Inconclusive", report)
        self.assertIn("connection refused", report)
        suggest.assert_not_called()


class LocalModeRegressionTests(unittest.TestCase):
    def test_loopback_target_still_uses_the_suggester(self):
        client = FakeClient(_script(LOCAL_URL, LOCAL_ANALYSIS))

        with patch("agent.run_full_diagnostic", return_value=LOCAL_RESULT), patch(
            "agent.source_file_for", return_value=None
        ), patch("agent.load_source_context", return_value=""), patch(
            "agent.suggest_code_fix", return_value="Use fewer workers."
        ) as suggest:
            report = _run_agent(f"Diagnose {LOCAL_URL} with 12 requests", client)

        self.assertEqual(client.calls[0]["messages"][0]["content"], agent.SYSTEM_PROMPT)
        self.assertIn("**Diagnosis**", report)
        self.assertIn("## Suggested fix", report)
        self.assertIn("Use fewer workers.", report)
        suggest.assert_called_once()

    def test_unknown_mode_is_rejected(self):
        client = FakeClient(lambda kwargs: None)

        with self.assertRaises(ValueError):
            agent.run_agent("anything", client=client, mode="x")


if __name__ == "__main__":
    unittest.main()
