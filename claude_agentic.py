"""Stage 3 — true two-agent Claude debate for paper evaluation."""

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

_REQUIRED_VERDICT_KEYS: frozenset[str] = frozenset({
    "technical_founder",
    "accelerator_partner",
    "final_verdict",
})

# --- Agent 1: Technical Founder ---

_TF_SYSTEM = (
    "You are a Technical Founder / Staff Research Engineer evaluating ML papers "
    "for startup potential. Assess technical feasibility, novelty, and defensible "
    "technical moat. Reply ONLY with valid JSON: "
    '{\"score\": <int 1-5>, \"rationale\": \"<1-2 sentences>\"}'
)

# --- Agent 2: Accelerator Partner ---

_AP_SYSTEM = (
    "You are a Deep-Tech Accelerator Partner evaluating ML papers for startup "
    "potential. Assess market pull, fundability, and accelerator story. A Technical "
    "Founder has already reviewed this paper — read their view, then give your own "
    "assessment AND a final converged verdict. Reply ONLY with valid JSON following "
    "this schema:\n"
    "{\n"
    '  "accelerator_partner": {"score": <int 1-5>, "rationale": "<1-2 sentences>"},\n'
    '  "final_verdict": {"score": <int 1-5>, "label": "keep|maybe|drop", "reason": "<1-2 sentences>"}\n'
    "}"
)

_RATIONALE_MAX_LEN = 500


def _truncate(text: str, max_len: int = _RATIONALE_MAX_LEN) -> str:
    """Truncate text to max_len characters, appending '…' if clipped."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def normalize_claude_verdict(raw: dict[str, Any]) -> dict[str, Any]:
    """Guarantee a well-typed, fully-populated verdict dict.

    - Fills any missing top-level or nested keys with safe defaults.
    - Coerces scores to int where possible; uses None on failure.
    - Truncates rationale / reason strings to _RATIONALE_MAX_LEN chars.

    Safe to call on both real parsed verdicts and the placeholder.
    """

    def _coerce_score(val: Any) -> int | None:
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _coerce_str(val: Any) -> str:
        s = val.strip() if isinstance(val, str) else ""
        return _truncate(s)

    tf_raw = raw.get("technical_founder") or {}
    ap_raw = raw.get("accelerator_partner") or {}
    fv_raw = raw.get("final_verdict") or {}

    return {
        "technical_founder": {
            "score": _coerce_score(tf_raw.get("score")),
            "rationale": _coerce_str(tf_raw.get("rationale")),
        },
        "accelerator_partner": {
            "score": _coerce_score(ap_raw.get("score")),
            "rationale": _coerce_str(ap_raw.get("rationale")),
        },
        "final_verdict": {
            "score": _coerce_score(fv_raw.get("score")),
            "label": str(fv_raw.get("label") or "unknown").strip(),
            "reason": _coerce_str(fv_raw.get("reason")),
        },
    }


def _paper_context(paper: Paper, openai_score: dict[str, Any]) -> str:
    return (
        f"Paper to evaluate:\n"
        f"Title: {paper.title}\n"
        f"URL: {paper.url}\n"
        f"Abstract: {paper.abstract or 'Not available.'}\n\n"
        f"OpenAI preliminary scores:\n{json.dumps(openai_score, indent=2)}"
    )


def debate_paper_with_two_agents(
    paper: Paper, openai_score: dict[str, Any]
) -> dict[str, Any]:
    """Run a true two-agent Claude debate and return a normalized verdict dict.

    Makes two sequential API calls — one per agent — so each agent responds
    independently. Gracefully returns a placeholder dict if ANTHROPIC_API_KEY
    is not set so the pipeline continues without Claude configured.

    The returned dict always has the full structure guaranteed by
    normalize_claude_verdict (all keys present, scores are int | None).
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        LOGGER.warning(
            "Claude debate skipped for paper_id=%s: no ANTHROPIC_API_KEY configured.",
            paper.paper_id,
        )
        return normalize_claude_verdict(dict(_PLACEHOLDER_VERDICT))

    from anthropic_client import claude_chat  # noqa: PLC0415 — lazy import

    context = _paper_context(paper, openai_score)

    last_error: Exception | None = None
    for attempt in range(1, 3):
        try:
            # Call 1 — Technical Founder
            tf_messages = [
                {"role": "system", "content": _TF_SYSTEM},
                {"role": "user", "content": context},
            ]
            raw_tf = claude_chat(tf_messages, max_tokens=256)
            tf_result = _parse_agent_response(raw_tf, required_keys={"score", "rationale"})
            LOGGER.info(
                "TF agent done for paper_id=%s: score=%s",
                paper.paper_id,
                tf_result.get("score"),
            )

            # Call 2 — Accelerator Partner (sees TF's view)
            ap_user = (
                f"{context}\n\n"
                f"Technical Founder's assessment:\n{json.dumps(tf_result, indent=2)}"
            )
            ap_messages = [
                {"role": "system", "content": _AP_SYSTEM},
                {"role": "user", "content": ap_user},
            ]
            raw_ap = claude_chat(ap_messages, max_tokens=512)
            ap_result = _parse_agent_response(
                raw_ap, required_keys={"accelerator_partner", "final_verdict"}
            )
            LOGGER.info(
                "AP agent done for paper_id=%s: label=%s score=%s",
                paper.paper_id,
                ap_result.get("final_verdict", {}).get("label"),
                ap_result.get("final_verdict", {}).get("score"),
            )

            raw_verdict: dict[str, Any] = {
                "technical_founder": tf_result,
                "accelerator_partner": ap_result["accelerator_partner"],
                "final_verdict": ap_result["final_verdict"],
            }
            return normalize_claude_verdict(raw_verdict)

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
    return normalize_claude_verdict(dict(_PLACEHOLDER_VERDICT))


def _parse_agent_response(content: str, required_keys: set[str]) -> dict[str, Any]:
    """Parse a Claude agent's JSON response; falls back to scanning for first { block."""
    try:
        parsed = json.loads(content)
    except JSONDecodeError:
        parsed = _extract_first_json_object(content)

    if not isinstance(parsed, dict):
        raise RuntimeError("Expected JSON object from Claude agent response")

    missing = required_keys - parsed.keys()
    if missing:
        raise RuntimeError(f"Claude agent response missing required keys: {missing}")

    return parsed


def _parse_verdict(content: str) -> dict[str, Any]:
    """Parse a full verdict JSON; used in unit tests and legacy callers."""
    return _parse_agent_response(content, required_keys=_REQUIRED_VERDICT_KEYS)


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
