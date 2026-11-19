"""Phase 5 tests: stage the fix, re-measure it, put everything back.

The loop is the first component that starts and stops servers and writes files, so
these tests pin the parts that are dangerous if wrong: the staged file is
validated before it can ever be served, the measured file is never written, the
throwaway copy is always deleted, and the original server is restarted even when
the re-test blows up. Model and server are replaced by fakes, so a test run never
opens a port, spends tokens, or depends on host load.
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent
import feedback_loop
import suggester

ORIGINAL_SOURCE = (
    "from fastapi import FastAPI\n"
    "\n"
    "app = FastAPI()\n"
    "\n"
    '@app.get("/heavy")\n'
    "def heavy():\n"
    "    return {'status': 'ok'}\n"
)

STAGED_SOURCE = ORIGINAL_SOURCE.replace("def heavy():", "async def heavy():")

FIX = "Offload the work.\n\n```python\nasync def heavy():\n    ...\n```"

BEFORE = {
    "url": "http://127.0.0.1:8000/heavy",
    "total_requests": 12,
    "load_test": {
        "successful": 12,
        "failed": 0,
        "avg_latency_seconds": 1.5,
        "p95_latency_seconds": 1.6,
        "requests_per_second": 6.5,
    },
    "cpu_delta_percent": 40.0,
    "system_metrics": {"peak_cpu_percent": 98.0},
}

AFTER = {
    "url": "http://127.0.0.1:8000/heavy",
    "total_requests": 12,
    "load_test": {
        "successful": 12,
        "failed": 0,
        "avg_latency_seconds": 1.0,
        "p95_latency_seconds": 1.1,
        "requests_per_second": 9.8,
    },
    "cpu_delta_percent": 45.0,
    "system_metrics": {"peak_cpu_percent": 99.0},
}


def _message(content=None):
    return SimpleNamespace(content=content, tool_calls=None)


def _response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Scripted stand-in for `openai.OpenAI`, capturing the request it received."""

    def __init__(self, reply):
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return _response(_message(content=reply))

        self.chat = SimpleNamespace(completions=_Completions())


class FakeServer:
    """Records start/stop calls instead of binding a port."""

    def __init__(self, port=8000):
        self.port = port
        self.calls = []

    def stop(self):
        self.calls.append("stop")

    def start(self, module):
        self.calls.append(f"start:{module}")


class StagedFilePromptTests(unittest.TestCase):
    def test_prompt_carries_the_fix_and_the_whole_file(self):
        client = FakeClient(f"```python\n{STAGED_SOURCE}```")

        staged = suggester.suggest_staged_file(
            ORIGINAL_SOURCE, FIX, file_name="dummy_api.py", client=client
        )

        prompt = client.calls[0]["messages"][1]["content"]
        self.assertIn(FIX.strip(), prompt)
        self.assertIn(ORIGINAL_SOURCE.strip(), prompt)
        self.assertIn("Return the ENTIRE modified file", prompt)
        self.assertIn("dummy_api.py", prompt)
        self.assertEqual(staged, STAGED_SOURCE)

    def test_fence_without_language_is_accepted(self):
        client = FakeClient(f"```\n{STAGED_SOURCE}```")

        staged = suggester.suggest_staged_file(ORIGINAL_SOURCE, FIX, client=client)

        self.assertTrue(staged.startswith("from fastapi import FastAPI"))

    def test_prose_around_the_block_is_ignored(self):
        client = FakeClient(f"Here you go:\n```python\n{STAGED_SOURCE}```\nDone.")

        staged = suggester.suggest_staged_file(ORIGINAL_SOURCE, FIX, client=client)

        self.assertNotIn("Here you go", staged)

    def test_missing_fix_is_rejected(self):
        with self.assertRaises(ValueError):
            suggester.suggest_staged_file(ORIGINAL_SOURCE, "   ")

    def test_empty_reply_is_an_error(self):
        client = FakeClient(None)

        with self.assertRaises(RuntimeError):
            suggester.suggest_staged_file(ORIGINAL_SOURCE, FIX, client=client)


class ValidationTests(unittest.TestCase):
    def test_valid_replacement_passes(self):
        self.assertEqual(
            feedback_loop.validate_staged_source(STAGED_SOURCE, "/heavy"),
            STAGED_SOURCE,
        )

    def test_empty_file_is_rejected(self):
        with self.assertRaises(ValueError):
            feedback_loop.validate_staged_source("   ")

    def test_diff_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            feedback_loop.validate_staged_source("--- a/api.py\n+++ b/api.py\n")

        self.assertIn("diff", str(caught.exception))

    def test_syntax_error_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            feedback_loop.validate_staged_source("app = FastAPI(\n")

        self.assertIn("does not parse", str(caught.exception))

    def test_missing_app_object_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            feedback_loop.validate_staged_source("handler = lambda: None\n")

        self.assertIn("`app`", str(caught.exception))

    def test_dropped_route_is_rejected(self):
        without_route = ORIGINAL_SOURCE.replace('@app.get("/heavy")', '@app.get("/x")')

        with self.assertRaises(ValueError) as caught:
            feedback_loop.validate_staged_source(without_route, "/heavy")

        self.assertIn("/heavy", str(caught.exception))

    def test_route_check_is_optional(self):
        feedback_loop.validate_staged_source(STAGED_SOURCE)


class CompareMetricsTests(unittest.TestCase):
    def test_faster_staged_run_is_improved(self):
        result = feedback_loop.compare_metrics(BEFORE, AFTER)

        self.assertEqual(result["verdict"], "improved")
        self.assertAlmostEqual(result["improvement_percent"], 33.33, places=2)
        self.assertEqual(result["changes"]["failed_requests"], 0)

    def test_change_inside_the_noise_band_is_not_a_win(self):
        after = {
            **AFTER,
            "load_test": {**AFTER["load_test"], "avg_latency_seconds": 1.45},
        }

        result = feedback_loop.compare_metrics(BEFORE, after)

        self.assertEqual(result["verdict"], "no material change")
        self.assertIn("noise threshold", result["reason"])

    def test_slower_staged_run_regresses(self):
        after = {
            **AFTER,
            "load_test": {**AFTER["load_test"], "avg_latency_seconds": 2.0},
        }

        result = feedback_loop.compare_metrics(BEFORE, after)

        self.assertEqual(result["verdict"], "regressed")
        self.assertIn("rose", result["reason"])

    def test_new_failures_beat_a_latency_win(self):
        after = {
            **AFTER,
            "load_test": {
                **AFTER["load_test"],
                "avg_latency_seconds": 0.5,
                "failed": 2,
                "successful": 10,
            },
        }

        result = feedback_loop.compare_metrics(BEFORE, after)

        self.assertEqual(result["verdict"], "regressed")
        self.assertIn("failed request", result["reason"])

    def test_staged_run_without_successes_is_inconclusive(self):
        after = {
            **AFTER,
            "load_test": {**AFTER["load_test"], "successful": 0, "failed": 12},
        }

        result = feedback_loop.compare_metrics(BEFORE, after)

        self.assertEqual(result["verdict"], "inconclusive")

    def test_missing_baseline_is_inconclusive(self):
        result = feedback_loop.compare_metrics({}, AFTER)

        self.assertEqual(result["verdict"], "inconclusive")

    def test_staged_run_with_no_metrics_is_not_blamed_on_failures(self):
        # A dropped payload and a run where every request failed are different
        # bugs, so they must not share a reason string.
        result = feedback_loop.compare_metrics(BEFORE, {})

        self.assertEqual(result["verdict"], "inconclusive")
        self.assertIn("no load-test metrics", result["reason"])

    def test_report_states_the_verdict_and_both_runs(self):
        comparison = feedback_loop.compare_metrics(BEFORE, AFTER)

        report = feedback_loop.format_comparison(
            comparison,
            url=BEFORE["url"],
            original_file="dummy_api.py",
            total_requests=12,
            staged_file="staging_api.py",
        )

        self.assertIn("**Verdict: improved**", report)
        self.assertIn("dummy_api.py` was never modified", report)
        self.assertIn("1500 ms", report)
        self.assertIn("1000 ms", report)
        # Change is stated after-vs-before, so a faster run is a negative change.
        self.assertIn("-33.3%", report)
        self.assertIn("| +0 |", report)

    def test_slower_run_reads_as_a_positive_latency_change(self):
        after = {
            **AFTER,
            "load_test": {**AFTER["load_test"], "avg_latency_seconds": 3.0},
        }

        report = feedback_loop.format_comparison(
            feedback_loop.compare_metrics(BEFORE, after)
        )

        self.assertIn("**Verdict: regressed**", report)
        self.assertIn("+100.0%", report)



class FeedbackLoopTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.source = self.root / "dummy_api.py"
        self.source.write_text(ORIGINAL_SOURCE, encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, **overrides):
        server = overrides.pop("server", FakeServer())
        settings = {
            "rewrite": lambda source, fix: STAGED_SOURCE,
            "load": lambda url, total: AFTER,
            "server": server,
            "project_root": self.root,
        }
        settings.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            result = feedback_loop.run_feedback_loop(
                "http://127.0.0.1:8000/heavy", self.source, FIX, BEFORE, "", **settings
            )
        return result, server

    def _run_with_path(self, path, server):
        return feedback_loop.run_feedback_loop(
            "http://127.0.0.1:8000/heavy",
            path,
            FIX,
            BEFORE,
            "",
            rewrite=lambda source, fix: STAGED_SOURCE,
            load=lambda url, total: AFTER,
            server=server,
            project_root=self.root,
        )

    def test_happy_path_serves_the_copy_then_restores_the_original(self):
        result, server = self._run()

        self.assertEqual(result["comparison"]["verdict"], "improved")
        self.assertEqual(result["staged_source"], STAGED_SOURCE)
        self.assertEqual(
            server.calls,
            ["stop", "start:staging_api", "stop", "start:dummy_api"],
        )
        self.assertTrue(result["restored"])
        self.assertIsNone(result["restore_error"])
        # The throwaway copy is gone and the measured file is byte-identical.
        self.assertFalse((self.root / "staging_api.py").exists())
        self.assertEqual(self.source.read_text(encoding="utf-8"), ORIGINAL_SOURCE)

    def test_failed_retest_still_cleans_up_and_restores(self):
        def explode(url, total):
            raise RuntimeError("the staged server returned 500s")

        server = FakeServer()
        with self.assertRaises(RuntimeError) as caught:
            self._run(server=server, load=explode)

        self.assertIn("the staged server returned 500s", str(caught.exception))
        self.assertFalse((self.root / "staging_api.py").exists())
        self.assertEqual(self.source.read_text(encoding="utf-8"), ORIGINAL_SOURCE)
        self.assertEqual(
            server.calls,
            ["stop", "start:staging_api", "stop", "start:dummy_api"],
        )

    def test_invalid_rewrite_never_starts_a_server(self):
        server = FakeServer()
        with self.assertRaises(ValueError):
            self._run(server=server, rewrite=lambda source, fix: "print('hi')\n")

        self.assertEqual(server.calls, [])
        self.assertFalse((self.root / "staging_api.py").exists())

    def test_preexisting_staging_file_blocks_the_run(self):
        (self.root / "staging_api.py").write_text("app = object()\n", encoding="utf-8")
        server = FakeServer()

        with self.assertRaises(RuntimeError) as caught:
            self._run(server=server)

        self.assertIn("already exists", str(caught.exception))
        self.assertEqual(server.calls, [])

    def test_missing_source_file_is_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(FileNotFoundError):
                self._run_with_path(self.root / "nope.py", FakeServer())


class PortTests(unittest.TestCase):
    def test_explicit_port_is_used(self):
        self.assertEqual(
            feedback_loop.port_from_url("http://127.0.0.1:8123/heavy"), 8123
        )

    def test_port_less_url_falls_back_to_the_uvicorn_default(self):
        # A target with no port is still assumed to be the local uvicorn dev
        # server (8000), not :80: the loop restarts a dev server, not a front end.
        self.assertEqual(
            feedback_loop.port_from_url("http://example.test/heavy"),
            feedback_loop.DEFAULT_PORT,
        )
        self.assertEqual(
            feedback_loop.port_from_url("https://example.test/heavy"), 443
        )


class VerifyPromptTests(unittest.TestCase):
    def test_non_interactive_run_skips_with_a_hint(self):
        class _NotATerminal(io.StringIO):
            def isatty(self):
                return False

        output = io.StringIO()
        with patch("sys.stdin", _NotATerminal()), contextlib.redirect_stdout(output):
            self.assertFalse(agent._ask_to_verify())

        self.assertIn("--verify", output.getvalue())

    def test_yes_on_a_terminal_accepts(self):
        class _Terminal(io.StringIO):
            def isatty(self):
                return True

        with patch("sys.stdin", _Terminal()), patch(
            "builtins.input", return_value="y"
        ):
            self.assertTrue(agent._ask_to_verify())

    def test_anything_else_declines(self):
        class _Terminal(io.StringIO):
            def isatty(self):
                return True

        with patch("sys.stdin", _Terminal()), patch(
            "builtins.input", return_value="no"
        ):
            self.assertFalse(agent._ask_to_verify())


if __name__ == "__main__":
    unittest.main()

