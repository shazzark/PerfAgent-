"""Phase 7 tests: the ruthless ramp-up that runs until the API is dead.

The loop is tested with scripted step functions: one API serves low rates and
raises `ConnectionError` at 40 req/s, another returns HTTP 500s for every
request, and a third stays healthy but slow. The scripts pin the stop rules, the
call sequence, and the payload shape without real sockets or host-dependent
timing. Separate tests cover the burst aggregator and the agent wiring.
"""

import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests

import agent
import tools

URL = "http://api.test/slow"


def _metrics(offered_rps, failed=0, avg_latency=0.4, failure_kind="HTTP 500"):
    """Build the subset of a ramp-step payload the loop consumes."""
    total = offered_rps
    success = total - failed
    breakdown = {failure_kind: failed} if failed else {}
    return {
        "url": URL,
        "offered_rps": offered_rps,
        "burst_seconds": 1.0,
        "total_requests": total,
        "successful": success,
        "failed": failed,
        "failure_rate": failed / total,
        "failure_rate_percent": round(failed / total * 100, 2),
        "failure_breakdown": breakdown,
        "connection_failures": sum(
            count
            for kind, count in breakdown.items()
            if kind.startswith("connection_")
        ),
        "all_failed": total > 0 and success == 0,
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


def _fails_at(rps, failure_kind="HTTP 500"):
    """A mock API that serves everything below `rps` and fails from `rps` up."""

    def step(url, offered_rps):
        failed = offered_rps if offered_rps >= rps else 0
        return _metrics(offered_rps, failed=failed, failure_kind=failure_kind)

    return step


class FindBreakingPointTests(unittest.TestCase):
    def test_total_failure_breaks_at_30_rps(self):
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
        self.assertEqual(result["final_metrics"]["failed"], 30)
        self.assertEqual(result["final_metrics"]["failure_rate_percent"], 100.0)
        self.assertEqual(result["crash_reason"], "100% HTTP 500s")
        self.assertEqual(result["crash_reason"], result["breaking_reason"])
        self.assertIs(result["final_metrics"], result["breaking_point"])

    def test_connection_error_at_40_rps_stops_the_ruthless_ramp(self):
        """The API survives 10 and 20 req/s, then refuses connections at 40."""

        def step(url, offered_rps):
            if offered_rps >= 40:
                raise requests.exceptions.ConnectionError("Connection refused")
            return _metrics(offered_rps)

        with patch("tools._run_ramp_step", side_effect=step) as run:
            result = tools.find_breaking_point(URL, start_rps=10)

        self.assertEqual(
            [call.args for call in run.call_args_list],
            [(URL, 10), (URL, 20), (URL, 40)],
        )
        self.assertTrue(result["found"])
        self.assertEqual(result["breaking_rps"], 40)
        self.assertEqual(result["last_successful_rps"], 20)
        self.assertEqual(result["crash_reason"], "Connection Refused")
        self.assertEqual(result["final_metrics"]["offered_rps"], 40)
        self.assertEqual(result["strategy"], "double")
        self.assertIn("Connection refused", result["final_metrics"]["error"])

    def test_partial_failures_do_not_stop_the_ramp(self):
        """A 50% failure rate is recorded, then the ramp keeps climbing."""

        def step(url, offered_rps):
            return _metrics(offered_rps, failed=offered_rps // 2)

        with patch("tools._run_ramp_step", side_effect=step) as run:
            result = tools.find_breaking_point(URL, start_rps=10, max_rps=80)

        self.assertEqual(
            [call.args[1] for call in run.call_args_list], [10, 20, 40, 80]
        )
        self.assertFalse(result["found"])
        self.assertIsNone(result["crash_reason"])
        self.assertEqual(result["last_successful_rps"], 80)
        self.assertEqual(result["final_metrics"]["failure_rate_percent"], 50.0)

    def test_slow_responses_do_not_stop_the_ramp(self):
        """Multi-second latency is not a crash; the ramp must keep going."""

        def step(url, offered_rps):
            return _metrics(offered_rps, avg_latency=5.0)

        with patch("tools._run_ramp_step", side_effect=step) as run:
            result = tools.find_breaking_point(URL, start_rps=10, max_rps=40)

        self.assertEqual(
            [call.args[1] for call in run.call_args_list], [10, 20, 40]
        )
        self.assertFalse(result["found"])
        self.assertIsNone(result["crash_reason"])
        self.assertEqual(result["last_successful_rps"], 40)

    def test_doubling_is_the_default_strategy(self):
        with patch("tools._run_ramp_step", side_effect=_fails_at(5000)) as step:
            result = tools.find_breaking_point(URL, start_rps=10, max_rps=1000)

        self.assertEqual(
            [call.args[1] for call in step.call_args_list],
            [10, 20, 40, 80, 160, 320, 640, 1000],
        )
        self.assertFalse(result["found"])
        self.assertEqual(result["last_successful_rps"], 1000)

    def test_surviving_max_rps_reports_no_break(self):
        with patch("tools._run_ramp_step", side_effect=_fails_at(1000)):
            result = tools.find_breaking_point(
                URL, start_rps=10, step_increase=10, max_rps=30
            )

        self.assertFalse(result["found"])
        self.assertIsNone(result["breaking_rps"])
        self.assertIsNone(result["crash_reason"])
        self.assertIsNone(result["breaking_reason"])
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
            {"step_increase": True},
            {"step_increase": 1.5},
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
        outcomes = [(True, 1.0, 0.0, None)] * 8 + [
            (False, 3.0, 0.0, "HTTP 500")
        ] * 2

        result = tools._summarize_ramp_burst(URL, 10, outcomes, elapsed=2.5)

        self.assertEqual(result["total_requests"], 10)
        self.assertEqual(result["successful"], 8)
        self.assertEqual(result["failed"], 2)
        self.assertEqual(result["failure_rate_percent"], 20.0)
        self.assertEqual(result["failure_breakdown"], {"HTTP 500": 2})
        self.assertEqual(result["connection_failures"], 0)
        self.assertFalse(result["all_failed"])
        self.assertEqual(result["avg_latency_seconds"], 1.4)
        self.assertEqual(result["end_to_end_avg_latency_seconds"], 1.4)
        self.assertEqual(result["p95_latency_seconds"], 3.0)
        self.assertEqual(result["max_latency_seconds"], 3.0)

    def test_summary_flags_a_connection_refusal_as_a_total_failure(self):
        outcomes = [(False, 0.01, 0.0, "connection_refused")] * 10

        result = tools._summarize_ramp_burst(URL, 10, outcomes, elapsed=1.0)

        self.assertEqual(result["connection_failures"], 10)
        self.assertEqual(result["failure_breakdown"], {"connection_refused": 10})
        self.assertTrue(result["all_failed"])

    def test_client_queueing_is_flagged_separately(self):
        # Twenty requests through a ten-worker pool: half waited for a worker.
        outcomes = [(True, 1.0, 0.0, None)] * 10 + [(True, 1.0, 1.0, None)] * 10

        result = tools._summarize_ramp_burst(URL, 20, outcomes, elapsed=2.0)

        self.assertTrue(result["client_limited"])
        self.assertEqual(result["client_queue_avg_seconds"], 0.5)
        self.assertEqual(result["client_queue_max_seconds"], 1.0)

    def test_burst_sends_one_request_per_offered_rps(self):
        with patch(
            "tools._ramp_request", return_value=(True, 0.5, 0.0, None)
        ) as request, patch("tools.time.sleep"):
            result = tools._run_ramp_step(URL, 10)

        self.assertEqual(request.call_count, 10)
        self.assertEqual(result["offered_rps"], 10)
        self.assertEqual(result["total_requests"], 10)
        self.assertEqual(result["successful"], 10)
        self.assertEqual(result["failure_rate_percent"], 0.0)
        self.assertEqual(result["avg_latency_seconds"], 0.5)

    def test_request_classifies_a_refused_connection(self):
        with patch(
            "tools.requests.get",
            side_effect=requests.exceptions.ConnectionError(
                "Connection refused by the target"
            ),
        ):
            succeeded, _latency, _wait, kind = tools._ramp_request(URL, 0.0)

        self.assertFalse(succeeded)
        self.assertEqual(kind, "connection_refused")

    def test_request_classifies_an_http_error(self):
        response = SimpleNamespace(status_code=500)
        with patch("tools.requests.get", return_value=response):
            succeeded, _latency, _wait, kind = tools._ramp_request(URL, 0.0)

        self.assertFalse(succeeded)
        self.assertEqual(kind, "HTTP 500")


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


CRASH_STEP = _metrics(20, failed=20, failure_kind="connection_refused")
CRASH_STEP["error"] = "Connection refused by the target"

RAMP_RESULT = {
    "url": URL,
    "start_rps": 10,
    "step_increase": 10,
    "max_rps": 30,
    "strategy": "linear",
    "found": True,
    "breaking_rps": 20,
    "last_successful_rps": 10,
    "crash_reason": "Connection Refused",
    "breaking_reason": "Connection Refused",
    "breaking_point": CRASH_STEP,
    "final_metrics": CRASH_STEP,
    "steps": [],
    "stop_conditions": {
        "stop_on_total_failure": True,
        "stop_on_connection_error": True,
        "failure_rate_percent": 100.0,
    },
    "client_concurrency_ceiling": 10,
    "break_was_client_limited": False,
}

ANALYSIS = (
    "The API went completely dead at 20 req/s: every connection was refused. "
    "The last surviving rate was 10 req/s, so the crash sits between 10 and "
    "20 req/s. Raise the ceiling and rerun to confirm the target never recovers, "
    "then check the server logs for the process that stopped listening."
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
            url=URL, start_rps=5, step_increase=None, max_rps=30
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

    def test_execute_tool_defaults_to_doubling_and_the_python_ceiling(self):
        with patch("agent.find_breaking_point", return_value={"ok": True}) as ramp:
            agent._execute_tool("find_breaking_point", {"url": URL})

        ramp.assert_called_once_with(
            url=URL, start_rps=10, step_increase=None, max_rps=1000
        )

    def test_tool_schema_exposes_ramp_defaults(self):
        schemas = {tool["function"]["name"]: tool for tool in agent.TOOLS}

        self.assertIn("find_breaking_point", schemas)
        function = schemas["find_breaking_point"]["function"]
        self.assertEqual(function["parameters"]["required"], ["url"])
        self.assertEqual(
            set(function["parameters"]["properties"]),
            {"url", "start_rps", "step_increase", "max_rps"},
        )
        # The description is the model's contract: it has to say that only total
        # failure counts and that the default ceiling needs a stronger generator.
        self.assertIn("100%", function["description"])
        self.assertIn("1000", function["description"])
        self.assertIn("k6", function["description"])


if __name__ == "__main__":
    unittest.main()
