"""Local staging feedback loop - Phase 5: The Feedback Loop.

Phase 4 proves a diagnosis and proposes a fix. This module refuses to take that
fix on trust. It copies the measured target to `staging_api.py`, rewrites the
copy with the fix applied, serves the copy on the same port, re-runs the identical
diagnostic, and reports before/after numbers.

Safety is the whole design constraint. Nothing here touches production, and it
does not even touch the file it started from:

* The measured file (`dummy_api.py`) is only ever *read*.
* The fix lands in a throwaway `staging_api.py` that is deleted in a `finally`.
* The staged copy must parse, assign `app`, and still declare the measured route
  before any server is started. A model reply that is a diff or a fragment is
  rejected instead of served.
* The loop only stops a process whose command line is a uvicorn server on the
  target port; anything else listening there aborts the run instead of being
  killed.
* The original module is restarted in a `finally`, even when the load test
  raises, so a failed verification cannot leave the workstation without a server.

Diff application is deliberately absent. Model-written diffs carry context drift,
invented hunks and wrong line counts, and a patch that half-applies is hard to
tell apart from a real regression. The model is asked for the complete file
instead, and the guardrail is `ast.parse` rather than a patch tool.

Usage (normally driven by `py agent.py --verify "..."`):
    from feedback_loop import run_feedback_loop
"""

import ast
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import psutil
import requests

from suggester import suggest_staged_file
from tools import run_full_diagnostic

__all__ = [
    "DEFAULT_PORT",
    "MATERIAL_CHANGE_PERCENT",
    "STAGING_FILE_NAME",
    "STAGING_MODULE",
    "LocalServer",
    "compare_metrics",
    "format_comparison",
    "port_from_url",
    "run_feedback_loop",
    "validate_staged_source",
]

# The copy is served as a real module, so its name must be importable by uvicorn.
STAGING_MODULE = "staging_api"
STAGING_FILE_NAME = f"{STAGING_MODULE}.py"

SERVER_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# Spawning a process and importing FastAPI is fast, but a cold machine can take a
# few seconds; waiting 30 s avoids a flaky run without hiding a real crash (the
# process exiting is detected immediately and reported with its log).
READY_TIMEOUT_SECONDS = 30.0
SHUTDOWN_TIMEOUT_SECONDS = 10.0
READY_POLL_SECONDS = 0.2
HEALTH_PATH = "/health"

# A change has to beat measurement noise before it is called an improvement. Two
# runs of the same burst on a shared host differ by a few percent, so anything
# inside this band is reported as "no material change" rather than as a win.
MATERIAL_CHANGE_PERCENT = 5.0

_PROJECT_ROOT = Path(__file__).resolve().parent


def port_from_url(url: str) -> int:
    """Return the port the measured URL was served on."""
    parsed = urlparse(url)
    if parsed.port:
        return parsed.port
    return 443 if parsed.scheme == "https" else DEFAULT_PORT


def _port_in_use(port: int, host: str = SERVER_HOST) -> bool:
    """True when something accepts TCP connections on `port`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0



class LocalServer:
    """Owns the single uvicorn process that serves the measured port.

    `start()` always launches the module itself, so the loop can serve
    `staging_api:app`, tear it down, and put the original module back. `stop()`
    either terminates the process this instance started or, for a server the user
    started by hand, the uvicorn process listening on the port.
    """

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        project_root: Path | str = _PROJECT_ROOT,
        ready_timeout: float = READY_TIMEOUT_SECONDS,
    ) -> None:
        self.port = port
        self.project_root = Path(project_root)
        self.ready_timeout = ready_timeout
        self.process: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self._log_handle = None

    # -- lifecycle --------------------------------------------------------
    def start(self, module: str) -> None:
        """Serve `module:app` on `port` and wait until it answers."""
        if self.process is not None:
            self.stop()
        if _port_in_use(self.port):
            raise RuntimeError(
                f"port {self.port} is still in use; refusing to start a second "
                "server on top of it"
            )

        self.log_path = (
            Path(tempfile.gettempdir()) / f"perfagent-{module}-{self.port}.log"
        )
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                f"{module}:app",
                "--host",
                SERVER_HOST,
                "--port",
                str(self.port),
            ],
            cwd=str(self.project_root),
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
        )
        self._wait_until_ready(module)

    def stop(self) -> None:
        """Stop the server on `port`, refusing to kill anything that is not uvicorn."""
        if self.process is not None:
            self._terminate_popen(self.process)
            self.process = None
        else:
            for process in self._listeners():
                self._terminate(process)
        self._close_log()
        self._wait_until_free()

    def _terminate_popen(self, popen: subprocess.Popen) -> None:
        """Stop a process this instance started, via its psutil handle."""
        try:
            process = psutil.Process(popen.pid)
        except psutil.NoSuchProcess:
            pass
        else:
            self._terminate(process)
        # Reap the handle even if the process was already gone, so the PID is not
        # left as a zombie on platforms that care.
        try:
            popen.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            popen.kill()
            popen.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)

    def _terminate(self, process: psutil.Process) -> None:
        # Stop children first: `uvicorn --reload` runs the real server in a child.
        for child in process.children(recursive=True):
            self._kill(child)
        self._kill(process)

    @staticmethod
    def _kill(process: psutil.Process) -> None:
        try:
            process.terminate()
            process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        except psutil.NoSuchProcess:
            pass
        except psutil.TimeoutExpired:
            process.kill()
            process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)


    # -- discovery --------------------------------------------------------
    def _listener_pids(self) -> set[int]:
        pids: set[int] = set()
        try:
            for conn in psutil.net_connections(kind="inet"):
                if (
                    conn.status == psutil.CONN_LISTEN
                    and conn.laddr
                    and conn.laddr.port == self.port
                    and conn.pid
                ):
                    pids.add(conn.pid)
        except (psutil.AccessDenied, OSError):
            # net_connections needs privileges on some Windows setups; the
            # command-line scan below is the fallback.
            pass

        if pids:
            return pids

        for process in psutil.process_iter(["pid", "cmdline"]):
            cmdline = " ".join(process.info.get("cmdline") or []).lower()
            if "uvicorn" in cmdline and str(self.port) in cmdline:
                pids.add(process.info["pid"])
        return pids

    def _listeners(self) -> list[psutil.Process]:
        listeners: list[psutil.Process] = []
        for pid in self._listener_pids():
            try:
                process = psutil.Process(pid)
                cmdline = " ".join(process.cmdline()).lower()
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied as exc:
                raise RuntimeError(
                    f"cannot inspect pid {pid} on port {self.port}, so it cannot "
                    f"be safely stopped: {exc}"
                ) from exc

            if "uvicorn" not in cmdline:
                raise RuntimeError(
                    f"refusing to stop pid {pid}: port {self.port} is held by "
                    f"{cmdline or 'an unknown process'}, which is not a uvicorn "
                    "server started for this run"
                )
            listeners.append(process)
        return listeners

    # -- readiness --------------------------------------------------------
    def _responds(self) -> bool:
        try:
            requests.get(
                f"http://{SERVER_HOST}:{self.port}{HEALTH_PATH}", timeout=1.0
            )
            return True
        except requests.ConnectionError:
            return False
        except requests.RequestException:
            # Any HTTP response, even 404, means the socket is serving requests.
            return True

    def _wait_until_ready(self, module: str) -> None:
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"uvicorn exited with code {self.process.returncode} while "
                    f"starting {module}: {self._log_tail()}"
                )
            if self._responds():
                print(f"✅ {module}:app is serving on port {self.port}")
                return
            time.sleep(READY_POLL_SECONDS)
        raise RuntimeError(
            f"{module}:app did not answer on port {self.port} within "
            f"{self.ready_timeout:.0f}s: {self._log_tail()}"
        )

    def _wait_until_free(self) -> None:
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if not _port_in_use(self.port):
                return
            time.sleep(READY_POLL_SECONDS)
        raise RuntimeError(
            f"port {self.port} is still accepting connections after shutdown"
        )

    def _log_tail(self, limit: int = 1200) -> str:
        if self.log_path is None:
            return "(no server log)"
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(server log unavailable)"
        text = text.strip()
        return text[-limit:] if text else "(server log is empty)"

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


# -- validation ------------------------------------------------------------
def validate_staged_source(source: str, route_path: str | None = None) -> str:
    """Return `source` if it is a runnable replacement, else raise ValueError.

    This is the gate between "a model wrote a file" and "a server executes it".
    It is intentionally cheap and structural - parse, `app` assignment, route
    still present - because those are the three ways a staged reply breaks a real
    service. It is not a linter and does not try to prove the fix is correct.

    Args:
        source: The complete staged file.
        route_path: The measured endpoint, e.g. ``/heavy``, which must survive the
            rewrite. ``None`` skips that check.

    Raises:
        ValueError: If the source is empty, is a diff, does not parse, does not
            assign ``app``, or dropped the measured route.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError("the staged file is empty")

    if "@@" in source or source.lstrip().startswith(("---", "+++")):
        raise ValueError(
            "the model returned a diff instead of a complete file; a partial "
            "patch is not safe to serve"
        )

    try:
        tree = ast.parse(source, filename=STAGING_FILE_NAME)
    except SyntaxError as exc:
        raise ValueError(
            f"the staged file does not parse: {exc.msg} (line {exc.lineno})"
        ) from exc

    defines_app = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "app"
            for target in node.targets
        )
        for node in tree.body
    )
    if not defines_app:
        raise ValueError(
            "the staged file never assigns `app`, so `staging_api:app` cannot "
            "serve it"
        )

    if route_path and route_path not in source:
        raise ValueError(
            f"the staged file no longer declares {route_path}; refusing to "
            "verify a fix that removed the endpoint under test"
        )
    return source



# -- comparison ------------------------------------------------------------
def _load_test(metrics: dict[str, Any] | None) -> dict[str, Any]:
    return (metrics or {}).get("load_test") or {}


def _change_percent(before: Any, after: Any) -> float | None:
    """Percent change from `before` to `after`, rounded, or None if unusable."""
    try:
        before_value = float(before)
        after_value = float(after)
    except (TypeError, ValueError):
        return None
    if before_value == 0:
        return None
    return round((after_value - before_value) / before_value * 100, 2)


def _improvement_percent(before: Any, after: Any) -> float | None:
    """Percent *reduction* - positive means the staged run was faster."""
    change = _change_percent(before, after)
    return None if change is None else round(-change, 2)


def compare_metrics(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Compare the original and staged diagnostic payloads.

    A win is a lower average latency that clears `MATERIAL_CHANGE_PERCENT`, with
    no new failed requests. Anything smaller is reported as "no material change"
    rather than dressed up as an improvement: on a shared host, one run cannot
    distinguish a 2% gain from noise.

    Returns:
        A dict with ``verdict`` (``improved``, ``regressed``,
        ``no material change`` or ``inconclusive``), ``reason``, the extracted
        ``before``/``after`` metrics, and the computed ``changes``.
    """
    before_metrics, after_metrics = _load_test(before), _load_test(after)
    result: dict[str, Any] = {
        "verdict": "inconclusive",
        "reason": "",
        "improvement_percent": None,
        "before": {},
        "after": {},
        "changes": {},
    }

    for label, metrics in (("before", before), ("after", after)):
        load = _load_test(metrics)
        result[label] = {
            "avg_latency_seconds": load.get("avg_latency_seconds"),
            "p95_latency_seconds": load.get("p95_latency_seconds"),
            "requests_per_second": load.get("requests_per_second"),
            "failed": load.get("failed"),
            "cpu_delta_percent": (metrics or {}).get("cpu_delta_percent"),
            "peak_cpu_percent": ((metrics or {}).get("system_metrics") or {}).get(
                "peak_cpu_percent"
            ),
        }

    if not before_metrics:
        result["reason"] = "the original run has no load-test metrics to compare"
        return result
    if not after_metrics:
        result["reason"] = "the staged run returned no load-test metrics"
        return result
    if not after_metrics.get("successful"):
        result["reason"] = "the staged run produced no successful requests"
        return result

    changes: dict[str, Any] = {
        # Plain after-vs-before percentages. The direction that means "better"
        # differs per metric, so the table states the change and the verdict
        # states the interpretation - a reader must not have to invert a sign.
        "avg_latency_change_percent": _change_percent(
            before_metrics.get("avg_latency_seconds"),
            after_metrics.get("avg_latency_seconds"),
        ),
        "p95_latency_change_percent": _change_percent(
            before_metrics.get("p95_latency_seconds"),
            after_metrics.get("p95_latency_seconds"),
        ),
        "throughput_change_percent": _change_percent(
            before_metrics.get("requests_per_second"),
            after_metrics.get("requests_per_second"),
        ),
        "failed_requests": int(after_metrics.get("failed") or 0)
        - int(before_metrics.get("failed") or 0),
        "cpu_delta_points": None,
    }

    before_delta = (before or {}).get("cpu_delta_percent")
    after_delta = (after or {}).get("cpu_delta_percent")
    if before_delta is not None and after_delta is not None:
        changes["cpu_delta_points"] = round(after_delta - before_delta, 2)

    result["changes"] = changes
    improvement = _improvement_percent(
        before_metrics.get("avg_latency_seconds"),
        after_metrics.get("avg_latency_seconds"),
    )
    result["improvement_percent"] = improvement

    if changes["failed_requests"] > 0:
        result["verdict"] = "regressed"
        result["reason"] = (
            f"the staged run added {changes['failed_requests']} failed request(s)"
        )
    elif improvement is None:
        result["reason"] = "average latency was missing from one of the runs"
    elif improvement >= MATERIAL_CHANGE_PERCENT:
        result["verdict"] = "improved"
        result["reason"] = (
            f"average latency fell {improvement:.1f}%, beyond the "
            f"{MATERIAL_CHANGE_PERCENT:.0f}% noise threshold"
        )
    elif improvement <= -MATERIAL_CHANGE_PERCENT:
        result["verdict"] = "regressed"
        result["reason"] = f"average latency rose {abs(improvement):.1f}%"
    else:
        result["verdict"] = "no material change"
        result["reason"] = (
            f"average latency moved {improvement:+.1f}%, inside the "
            f"{MATERIAL_CHANGE_PERCENT:.0f}% noise threshold"
        )

    return result



def _fmt_seconds(value: Any) -> str:
    try:
        return f"{float(value) * 1000:.0f} ms"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_number(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_change(value: Any, unit: str = "%") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.1f}{unit}"


def _fmt_count(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):+d}"


def format_comparison(
    comparison: dict[str, Any],
    *,
    url: str | None = None,
    original_file: str | None = None,
    staged_file: str = STAGING_FILE_NAME,
    total_requests: int | None = None,
) -> str:
    """Render `comparison` as the markdown section the agent returns.

    Plain text rather than a table-only report on purpose: the verdict is the
    part the user acts on, so it is stated in a sentence before the numbers.
    """
    before = comparison.get("before") or {}
    after = comparison.get("after") or {}
    changes = comparison.get("changes") or {}
    verdict = comparison.get("verdict", "inconclusive")
    reason = comparison.get("reason") or ""

    target = f"`{url}`" if url else "the measured endpoint"
    scope = (
        f"{total_requests} requests were"
        if total_requests
        else "the same burst was"
    )
    lines = [
        "## Staging verification",
        "",
        f"The fix was applied to a copy (`{staged_file}`), served locally, and "
        f"{scope} re-run against {target}. "
        + (
            f"`{original_file}` was never modified."
            if original_file
            else "The measured file was never modified."
        ),
        "",
        "| Metric | Before | After | Change (after vs before) |",
        "| --- | --- | --- | --- |",
        (
            f"| Avg latency | {_fmt_seconds(before.get('avg_latency_seconds'))} | "
            f"{_fmt_seconds(after.get('avg_latency_seconds'))} | "
            f"{_fmt_change(changes.get('avg_latency_change_percent'))} |"
        ),
        (
            f"| p95 latency | {_fmt_seconds(before.get('p95_latency_seconds'))} | "
            f"{_fmt_seconds(after.get('p95_latency_seconds'))} | "
            f"{_fmt_change(changes.get('p95_latency_change_percent'))} |"
        ),
        (
            f"| Throughput | {_fmt_number(before.get('requests_per_second'))} req/s | "
            f"{_fmt_number(after.get('requests_per_second'))} req/s | "
            f"{_fmt_change(changes.get('throughput_change_percent'))} |"
        ),
        (
            f"| Failed requests | {before.get('failed', 'n/a')} | "
            f"{after.get('failed', 'n/a')} | "
            f"{_fmt_count(changes.get('failed_requests'))} |"
        ),
        (
            f"| CPU delta | {_fmt_number(before.get('cpu_delta_percent'))} pts | "
            f"{_fmt_number(after.get('cpu_delta_percent'))} pts | "
            f"{_fmt_change(changes.get('cpu_delta_points'), ' pts')} |"
        ),
        "",
        f"**Verdict: {verdict}**" + (f" - {reason}." if reason else "."),
        "",
        "Lower latency and higher throughput are better, so negative latency and "
        "positive throughput changes are wins.",
        "",
        "One local run cannot separate a small change from host noise, so only a "
        f"change of {MATERIAL_CHANGE_PERCENT:.0f}% or more is called real.",
    ]
    return "\n".join(lines)



# -- orchestration ---------------------------------------------------------
def _join_error(previous: str | None, message: str) -> str:
    return f"{previous}; {message}" if previous else message


def _read_source(path: Path, source_code_context: str = "") -> str:
    """Read the file that will be staged.

    The on-disk copy wins over `source_code_context` because the orchestrator may
    have truncated that context for the prompt, and a truncated file cannot be
    rewritten into a runnable module. The context is only a fallback for a path
    that has become unreadable since it was attached.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        context = (source_code_context or "").strip()
        if context.startswith("# File:"):
            context = context.split("\n", 1)[1] if "\n" in context else ""
        if not context.strip():
            raise RuntimeError(
                f"cannot read {path} and no source context was supplied: {exc}"
            ) from exc
        return context


def run_feedback_loop(
    url: str,
    source_path: str | Path,
    suggested_fix: str,
    original_metrics: dict[str, Any],
    source_code_context: str = "",
    *,
    total_requests: int | None = None,
    client=None,
    rewrite: Callable[[str, str], str] | None = None,
    server: LocalServer | None = None,
    load: Callable[[str, int], dict[str, Any]] | None = None,
    project_root: Path | str | None = None,
) -> dict[str, Any]:
    """Apply `suggested_fix` to a staging copy and measure the result.

    Sequence: read the original (never write it) -> ask the model for the complete
    rewritten file -> validate it -> write `staging_api.py` -> stop the server ->
    serve the copy on the same port -> re-run the identical diagnostic -> compare
    -> delete the copy and restart the original module.

    Args:
        url: The endpoint that was measured, e.g. ``http://127.0.0.1:8000/heavy``.
        source_path: The file that declares it. Only this file is copied.
        suggested_fix: The remediation text from `suggester.suggest_code_fix`.
        original_metrics: The `run_full_diagnostic` payload the fix was written
            against; it supplies the "before" numbers.
        source_code_context: The context the orchestrator attached, used only if
            `source_path` cannot be read.
        total_requests: Burst size for the re-test; defaults to the original run.
        client: Optional pre-built LLM client, reused for the rewrite call.
        rewrite: Optional replacement for the model call, for tests.
        server: Optional `LocalServer` (or compatible object) for tests.
        load: Optional replacement for `run_full_diagnostic`, for tests.
        project_root: Directory the staged copy is written to; defaults to this
            project. Tests point it at a temporary directory.

    Returns:
        ``{"comparison", "after", "staged_source", "restored", "restore_error",
        "original_file", "staging_file", "total_requests", "url"}``.

    Raises:
        FileNotFoundError: If `source_path` does not exist.
        RuntimeError: If the staged file already exists, the rewrite cannot be
            validated, or the staged run could not be measured. The original
            server is restarted before the exception leaves this function.
    """
    measured_url = url
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"cannot stage {source}: file not found")

    total = int(total_requests or (original_metrics or {}).get("total_requests") or 12)
    route_path = urlparse(measured_url).path or "/"

    staging_path = Path(project_root or _PROJECT_ROOT) / STAGING_FILE_NAME
    if staging_path.exists():
        # Deleting or overwriting a file this run did not create could destroy
        # someone's work; refuse instead of guessing.
        raise RuntimeError(
            f"{staging_path.name} already exists in {staging_path.parent}; "
            "remove it so this run does not overwrite an unrelated file"
        )

    original_source = _read_source(source, source_code_context)
    apply_fix = rewrite or (
        lambda code, fix: suggest_staged_file(
            code, fix, file_name=source.name, client=client
        )
    )

    print("🧪 Asking the model for the complete file with the fix applied...")
    staged_source = validate_staged_source(
        apply_fix(original_source, suggested_fix), route_path
    )
    staging_path.write_text(staged_source, encoding="utf-8")
    print(
        f"📝 Staged {staging_path.name} ({len(staged_source)} chars); "
        f"{source.name} was not modified."
    )


    server = server or LocalServer(port=port_from_url(measured_url))
    load = load or run_full_diagnostic

    after: dict[str, Any] | None = None
    failure: Exception | None = None
    restore_error: str | None = None
    try:
        print(f"🛑 Stopping the server on port {server.port}...")
        server.stop()
        print(f"🚀 Starting {STAGING_MODULE}:app on port {server.port}...")
        server.start(STAGING_MODULE)
        print(f"📊 Re-running the {total}-request diagnostic against the staged fix...")
        after = load(measured_url, total)
    except Exception as exc:  # noqa: BLE001 - cleanup must run before re-raising
        failure = exc
    finally:
        try:
            server.stop()
        except Exception as exc:  # noqa: BLE001 - report, never mask the cause
            restore_error = _join_error(
                restore_error, f"could not stop the staging server: {exc}"
            )
        try:
            staging_path.unlink(missing_ok=True)
            print(f"🧹 Removed {staging_path.name}.")
        except OSError as exc:
            restore_error = _join_error(
                restore_error, f"could not remove {staging_path.name}: {exc}"
            )
        try:
            print(f"♻️  Restarting {source.stem}:app on port {server.port}...")
            server.start(source.stem)
        except Exception as exc:  # noqa: BLE001 - a dead port must be reported
            message = f"could not restart {source.stem}:app: {exc}"
            restore_error = _join_error(restore_error, message)
            print(f"❌ {message}")

    if failure is not None:
        raise RuntimeError(f"staging verification failed: {failure}") from failure
    if after is None:
        raise RuntimeError("staging verification produced no metrics")

    comparison = compare_metrics(original_metrics, after)
    if comparison["reason"]:
        print(f"🧾 Verdict: {comparison['verdict']} - {comparison['reason']}")
    else:
        print(f"🧾 Verdict: {comparison['verdict']}")
    if restore_error:
        print(f"⚠️  Cleanup problem: {restore_error}")

    return {
        "url": measured_url,
        "original_file": str(source),
        "staging_file": str(staging_path),
        "total_requests": total,
        "staged_source": staged_source,
        "after": after,
        "comparison": comparison,
        "restored": restore_error is None,
        "restore_error": restore_error,
    }

