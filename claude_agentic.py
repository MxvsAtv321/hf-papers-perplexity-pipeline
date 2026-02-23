"""Stage 3 — two-agent Claude debate for paper evaluation."""

from __future__ import annotations

import json
import logging
import os
from json import JSONDecodeError
from typing import Any

from models import Paper

LOGGER = logging.getLogger(__name__)

_PLACEHOLDER_VERDICT: dict[str, Any] = {
    "technical_founder": {"score": None, "rationale": "claude_not_configured"},
    "accelerator_partner": {"score": None, "rationale": "claude_not_configured"},
    "final_verdict": {
        "score": None,
        "label": "unknown",
        "reason": "Claude API not configured",
    },
}

_SYSTEM_PROMPT = (
    "You are simulating a conversation between two experts evaluating ML papers "
    "as startup ideas. You must output ONLY valid JSON following the specified "
    "schema. No prose outside JSON."
)

_VERDICT_SCHEMA = """{
  "technical_founder": {
    "score": <int 1-5>,
    "rationale": "<1-2 sentences>"
  },
  "accelerator_partner": {
    "score": <int 1-5>,
    "rationale": "<1-2 sentences>"
  },
  "final_verdict": {
    "score": <int 1-5>,
    "label": "keep | maybe | drop",
    "reason": "<1-2 sentences>"
  }
}"""

_REQUIRED_VERDICT_KEYS: frozenset[str] = frozenset({
    "technical_founder",
    "accelerator_partner",
    "final_verdict",
})


def debate_paper_with_two_agents(
    paper: Paper, openai_score: dict[str, Any]
) -> dict[str, Any]:
    """Run a two-agent Claude debate and return a structured verdict.

    Gracefully returns a placeholder dict if ANTHROPIC_API_KEY is not set
    so the pipeline continues without Claude configured.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        LOGGER.warning(
            "Claude debate skipped for paper_id=%s: no ANTHROPIC_API_KEY configured.",
            paper.paper_id,
        )
        return dict(_PLACEHOLDER_VERDICT)

    from anthropic_client import claude_chat  # noqa: PLC0415 — lazy import

    user_content = f"""Paper to evaluate:
Title: {paper.title}
URL: {paper.url}
Abstract: {paper.abstract or "Not available."}

OpenAI preliminary scores:
{json.dumps(openai_score, indent=2)}

Two experts will now evaluate this paper as a startup opportunity:

Agent 1 — Technical Founder / Staff Research Engineer:
  Focuses on technical feasibility, novelty, and defensible technical moat.

Agent 2 — Deep-Tech Accelerator Partner:
  Focuses on market pull, accelerator story, and fundability.

Each agent gives:
  - A score (int 1–5, where 5 = extremely strong)
  - A 1–2 sentence rationale

They then briefly converge on a final verdict (score + label + reason).

Required JSON output (fill in integer scores and short strings):
{_VERDICT_SCHEMA}

Output ONLY the JSON object. No prose before or after."""

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    last_error: Exception | None = None
    for attempt in range(1, 3):
        try:
            raw = claude_chat(messages, max_tokens=512)
            verdict = _parse_verdict(raw)
            LOGGER.info(
                "Claude debate succeeded for paper_id=%s: label=%s score=%s",
                paper.paper_id,
                verdict.get("final_verdict", {}).get("label"),
                verdict.get("final_verdict", {}).get("score"),
            )
            return verdict
        except Exception as exc:
            last_error = exc
            LOGGER.warning(
                "Claude debate attempt %s/2 failed for paper_id=%s: %s",
                attempt,
                paper.paper_id,
                exc,
            )

    LOGGER.error(
        "Claude debate ultimately failed for paper_id=%s: %s", paper.paper_id, last_error
    )
    return dict(_PLACEHOLDER_VERDICT)


def _parse_verdict(content: str) -> dict[str, Any]:
    """Parse Claude's JSON verdict; falls back to scanning for first { block."""
    try:
        parsed = json.loads(content)
    except JSONDecodeError:
        parsed = _extract_first_json_object(content)

    if not isinstance(parsed, dict):
        raise RuntimeError("Expected JSON object from Claude verdict")

    missing = _REQUIRED_VERDICT_KEYS - parsed.keys()
    if missing:
        raise RuntimeError(f"Claude verdict missing required keys: {missing}")

    return parsed


def _extract_first_json_object(content: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for i, char in enumerate(content):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(content[i:])
        except JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    raise RuntimeError("Could not extract JSON object from Claude output")
