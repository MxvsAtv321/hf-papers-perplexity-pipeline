"""Thin wrapper around the Anthropic Messages API."""

from __future__ import annotations

import logging
import os
from typing import Any

import anthropic

LOGGER = logging.getLogger(__name__)


def claude_chat(messages: list[dict[str, str]], max_tokens: int = 1024) -> str:
    """Call the Claude API and return the assistant reply as a string.

    Args:
        messages: List of message dicts with "role" and "content" keys.
                  A "system" role message is extracted and passed via the
                  Anthropic API's dedicated system= parameter.
        max_tokens: Hard cap on output tokens (keep low for JSON-only tasks).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is required")

    claude_model = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")
    client = anthropic.Anthropic(api_key=api_key)

    system: str | None = None
    filtered: list[dict[str, Any]] = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            filtered.append({"role": msg["role"], "content": msg["content"]})

    kwargs: dict[str, Any] = {
        "model": claude_model,
        "max_tokens": max_tokens,
        "messages": filtered,
    }
    if system:
        kwargs["system"] = system

    LOGGER.debug("Calling Claude model=%s max_tokens=%s", claude_model, max_tokens)
    response = client.messages.create(**kwargs)
    return response.content[0].text
