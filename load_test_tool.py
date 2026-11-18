"""Backwards-compatible entry point for the Phase 1 load-testing tool.

In Phase 2 the implementation moved to `tools.py` and became the function that
the LLM orchestrator calls. Import `run_load_test` from `tools` in new code;
this module only re-exports it so Phase 1 notes and scripts keep working.
"""

from tools import run_load_test

__all__ = ["run_load_test"]

if __name__ == "__main__":
    print("⚠️  load_test_tool.py is deprecated - use tools.py instead.\n")
    # Start dummy_api first:
    #   py -m uvicorn dummy_api:app --reload
    run_load_test("http://127.0.0.1:8000/slow", total_requests=20)
