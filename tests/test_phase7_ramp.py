"""Phase 7 tests: automatic ramp-up until the breaking point.

The loop is tested with a scripted step function that behaves like an API which
is healthy below 30 req/s and fails at 30 req/s. That keeps the threshold rules,
call sequence, and payload shape pinned without real sockets or host-dependent
timing. Separate tests cover the burst aggregator and the agent wiring.
"""

import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent
import tools

URL = "http://api.test/slow"


def _metrics(offered_rps, failed=0, avg_latency=0.4):
    """Build the subset of a ramp-step payload the loop consumes."""
    total = offered_rps
    success = total - failed
    return {
        "url": URL,
        "offered_rps": offered_rps,
        "burst_seconds": 1.0,
        "total_requests": total,
        "successful": success,
        "failed": failed,
        "failure_rate": failed / total,
        "failure_rate_percent": round(failed / total * 100, 2),
        "total_time_seconds": 1.0,
        "completion_rps": float(total),
        "avg_latency_seconds": avg_latency,
        "p95_latency_seconds": avg_latency,
        "max_latency_seconds": avg_latency,
        "end_to_end_avg_latency_seconds": avg_latency,
        "client_queue_avg_seconds": 0.0,
        "client_queue_max_seconds": 0.0,
        "client_concurrency_ceiling": tools.MAX_WORKERS,
        "client_limited": False,
    }


def _fails_at(rps):
    """A mock API that serves everything below `rps` and fails from `rps` up."""

    def step(url, offered_rps):
        failed = offered_rps if offered_rps >= rps else 0
        return _metrics(offered_rps, failed=failed)

    return step


class FindBreakingPointTests(unittest.TestCase):
    def test_failure_rate_breaks_at_30_rps(self):
        with patch("tools._run_ramp_step", side_effect=_fails_at(30)) as step:
            result = tools.find_breaking_point(
                URL, start_rps=10, step_increase=10, max_rps=50
            )

        self.assertEqual(
            [call.args for call in step.call_args_list],
            [(URL, 10), (URL, 20), (URL, 30)],
        )
        self.assertTrue(result["found"])
        self.assertEqual(result["breaking_rps"], 30)
        self.assertEqual(result["last_successful_rps"], 20)
        self.assertEqual(result["breaking_point"]["failed"], 30)
        self.assertEqual(result["breaking_point"]["failure_rate_percent"], 100.0)
        self.assertIn("failure rate", result["breaking_reason"])

    def test_average_latency_breaks_the_ramp(self):
        def latency_step(url, offered_rps):
            latency = 2.4 if offered_rps >= 25 else 0.4
            return _metrics(offered_rps, avg_latency=latency)

        with patch("tools._run_ramp_step", side_effect=latency_step):
            result = tools.find_breaking_point(
                URL, start_rps=10, step_increase=5, max_rps=30
            )

        self.assertEqual(result["breaking_rps"], 25)
        self.assertEqual(result["last_successful_rps"], 20)
        self.assertEqual(result["breaking_reason"].split()[0], "end-to-end")
        self.assertIn("average latency", result["breaking_reason"])

    def test_surviving_max_rps_reports_no_break(self):
        with patch("tools._run_ramp_step", side_effect=_fails_at(1000)):
            result = tools.find_breaking_point(
                URL, start_rps=10, step_increase=10, max_rps=30
            )

        self.assertFalse(result["found"])
        self.assertIsNone(result["breaking_rps"])
        self.assertIsNone(result["breaking_point"])
        self.assertEqual(result["last_successful_rps"], 30)
        self.assertEqual(
            [step["offered_rps"] for step in result["steps"]], [10, 20, 30]
        )

    def test_breaking_on_the_first_step_has_no_last_success(self):
        with patch("tools._run_ramp_step", side_effect=_fails_at(5)):
            result = tools.find_breaking_point(
                URL, start_rps=5, step_increase=10, max_rps=30
            )

        self.assertEqual(result["breaking_rps"], 5)
        self.assertIsNone(result["last_successful_rps"])

    def test_final_step_is_clamped_to_max_rps(self):
        with patch("tools._run_ramp_step", side_effect=_fails_at(30)) as step:
            result = tools.find_breaking_point(
                URL, start_rps=5, step_increase=10, max_rps=30
            )

        self.assertEqual(
            [call.args[1] for call in step.call_args_list], [5, 15, 25, 30]
        )
        self.assertEqual(result["breaking_rps"], 30)
        self.assertEqual(result["last_successful_rps"], 25)

    def test_invalid_arguments_are_rejected_before_any_traffic(self):
        cases = [
            {"start_rps": 0},
            {"start_rps": True},
            {"step_increase": 0},
            {"max_rps": 5, "start_rps": 10},
            {"start_rps": 1.5},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), patch(
                "tools._run_ramp_step"
            ) as step:
                with self.assertRaises(ValueError):
                    tools.find_breaking_point(URL, **kwargs)
                step.assert_not_called()


class RampBurstTests(unittest.TestCase):
    def test_summary_counts_failures_and_percentiles(self):
        outcomes = [(True, 1.0, 0.0)] * 8 + [(False, 3.0, 0.0)] * 2

        result = tools._summarize_ramp_burst(URL, 10, outcomes, elapsed=2.5)

        self.assertEqual(result["total_requests"], 10)
        self.assertEqual(result["successful"], 8)
        self.assertEqual(result["failed"], 2)
        self.assertEqual(result["failure_rate_percent"], 20.0)
        self.assertEqual(result["avg_latency_seconds"], 1.4)
        self.assertEqual(result["end_to_end_avg_latency_seconds"], 1.4)
        self.assertEqual(result["p95_latency_seconds"], 3.0)
        self.assertEqual(result["max_latency_seconds"], 3.0)

    def test_client_queueing_is_flagged_separately(self):
        # Twenty requests through a ten-worker pool: half waited for a worker.
        outcomes = [(True, 1.0, 0.0)] * 10 + [(True, 1.0, 1.0)] * 10

        result = tools._summarize_ramp_burst(URL, 20, outcomes, elapsed=2.0)

        self.assertTrue(result["client_limited"])
        self.assertEqual(result["client_queue_avg_seconds"], 0.5)
        self.assertEqual(result["client_queue_max_seconds"], 1.0)

    def test_burst_sends_one_request_per_offered_rps(self):
        with patch(
            "tools._timed_request_meta", return_value=(True, 0.5, 0.0)
        ) as request, patch("tools.time.sleep"):
            result = tools._run_ramp_step(URL, 10)

        self.assertEqual(request.call_count, 10)
        self.assertEqual(result["offered_rps"], 10)
        self.assertEqual(result["total_requests"], 10)
        self.assertEqual(result["successful"], 10)
        self.assertEqual(result["failure_rate_percent"], 0.0)
        self.assertEqual(result["avg_latency_seconds"], 0.5)


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
    def __init__(self, script):
        self.chat = SimpleNamespace(completions=_FakeCompletions(script))

    @property
    def calls(self):
        return self.chat.completions.calls


RAMP_RESULT = {
    "url": URL,
    "found": True,
    "breaking_rps": 20,
    "last_successful_rps": 15,
    "breaking_reason": "end-to-end average latency 2.250s exceeded 2.0s",
    "breaking_point": _metrics(20, avg_latency=2.25),
    "steps": [],
    "thresholds": {
        "failure_rate_percent": 5.0,
        "avg_latency_seconds": 2.0,
        "end_to_end_avg_latency_seconds": 2.0,
    },
    "client_concurrency_ceiling": 10,
    "break_was_client_limited": True,
}

ANALYSIS = (
    "The API broke at 20 req/s: average latency was 2.25 s, with zero failures. "
    "The last clean step was 15 req/s. The 10-worker generator was queueing, so "
    "20 req/s is a lower bound rather than proof of the server's capacity."
)


def _ramp_then_answer():
    def script(kwargs):
        saw_tool_result = any(
            isinstance(message, dict) and message.get("role") == "tool"
            for message in kwargs["messages"]
        )
        if saw_tool_result:
            return _response(_message(content=ANALYSIS))
        return _response(
            _message(
                tool_calls=[
                    _tool_call(
                        "find_breaking_point",
                        json.dumps(
                            {
                                "url": URL,
                                "start_rps": 5,
                                "max_rps": 30,
                            }
                        ),
                    )
                ]
            )
        )

    return script


class AgentRampWiringTests(unittest.TestCase):
    def test_ramp_analysis_is_returned_without_a_suggested_fix(self):
        client = FakeClient(_ramp_then_answer())

        with patch(
            "agent.find_breaking_point", return_value=RAMP_RESULT
        ) as ramp, patch("agent.suggest_code_fix") as suggest, contextlib.redirect_stdout(
            io.StringIO()
        ):
            answer = agent.run_agent(
                "Find the breaking point for http://127.0.0.1:8000/slow "
                "from 5 to 30 RPS",
                client=client,
            )

        self.assertEqual(answer, ANALYSIS)
        ramp.assert_called_once_with(
            url=URL, start_rps=5, step_increase=10, max_rps=30
        )
        suggest.assert_not_called()

    def test_execute_tool_dispatches_ramp_arguments(self):
        with patch("agent.find_breaking_point", return_value={"ok": True}) as ramp:
            result = agent._execute_tool(
                "find_breaking_point",
                {
                    "url": URL,
                    "start_rps": 5,
                    "step_increase": 5,
                    "max_rps": 30,
                },
            )

        self.assertEqual(result, {"ok": True})
        ramp.assert_called_once_with(
            url=URL, start_rps=5, step_increase=5, max_rps=30
        )

    def test_tool_schema_exposes_ramp_defaults(self):
        schemas = {tool["function"]["name"]: tool for tool in agent.TOOLS}

        self.assertIn("find_breaking_point", schemas)
        parameters = schemas["find_breaking_point"]["function"]["parameters"]
        self.assertEqual(parameters["required"], ["url"])
        self.assertEqual(
            set(parameters["properties"]),
            {"url", "start_rps", "step_increase", "max_rps"},
        )


if __name__ == "__main__":
    unittest.main()
