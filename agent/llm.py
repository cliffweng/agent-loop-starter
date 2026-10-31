"""One HTTP call to an OpenAI-compatible chat endpoint.

No SDK: the loop only needs the assistant message, and the request body
is the whole wire format.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "x-ai/grok-4.3"


def load_local_env(path: str | None = None) -> None:
    """Load KEY=VALUE lines from .env. Variables already in the environment win."""
    env_path = Path(path) if path else Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def chat(messages, tools, model=None) -> dict:
    """POST /chat/completions and return the assistant message.

    The message has role, content, and tool_calls when the model wants a
    tool. tool_choice "auto" lets it either call a tool or answer in text.
    """
    load_local_env()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add "
            "your key, or export OPENROUTER_API_KEY."
        )

    base = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    chosen = model or os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
    response = requests.post(
        f"{base}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": chosen,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        },
        timeout=60,
    )
    # The body usually names the field OpenRouter rejected.
    if response.status_code >= 400:
        raise RuntimeError(
            f"LLM request failed ({response.status_code}): {response.text}"
        )
    try:
        return response.json()["choices"][0]["message"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Unexpected LLM response: {response.text}") from exc
