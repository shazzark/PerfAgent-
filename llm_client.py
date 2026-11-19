"""Shared LLM configuration and client factory.

Phase 4 added a second model call (the remediation suggester), so the endpoint,
model id, and key handling can no longer live inside `agent.py`: both the
orchestrator and `suggester.py` need the same client. Keeping the factory here
also avoids a circular import between those two modules.

`python-dotenv` reads `.env` at import. Real environment variables still win,
because `load_dotenv()` does not override values that are already set.

Usage:
    from llm_client import MODEL, create_client
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

__all__ = ["BASE_URL", "MODEL", "DEFAULT_BASE_URL", "DEFAULT_MODEL", "create_client"]

load_dotenv()

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

BASE_URL = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
MODEL = os.getenv("LLM_MODEL", DEFAULT_MODEL)


def create_client() -> OpenAI:
    """Build the Groq-backed OpenAI client, failing early on missing config."""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is not set. Put it in .env (see README.md, 'Configuration')."
        )
    return OpenAI(api_key=api_key, base_url=BASE_URL)
