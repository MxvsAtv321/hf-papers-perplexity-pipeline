"""Tests for claude_agentic.debate_paper_with_two_agents."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from claude_agentic import _parse_verdict, debate_paper_with_two_agents
from models import Paper

_SAMPLE_PAPER = Paper(
    paper_id="2501.00001",
    title="A New Agentic Framework",
    url="https://huggingface.co/papers/2501.00001",
    abstract="We propose a framework for agentic task planning.",
    published_at=datetime(2026, 2, 22, tzinfo=UTC),
)

_SAMPLE_SCORE = {
    "startup_potential": 4,
    "market_pull": 4,
    "technical_moat": 5,
    "story_for_accelerator": 4,
    "overall_score": 4,
    "rationale": "Strong technical moat.",
}

_VALID_VERDICT_JSON = """{
  "technical_founder": {"score": 4, "rationale": "Novel architecture."},
  "accelerator_partner": {"score": 5, "rationale": "Large TAM."},
  "final_verdict": {"score": 4, "label": "keep", "reason": "Worth investing."}
}"""


# ---------------------------------------------------------------------------
# No API key → placeholder returned, no exception raised
# ---------------------------------------------------------------------------

def test_debate_returns_placeholder_when_no_api_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        result = debate_paper_with_two_agents(_SAMPLE_PAPER, _SAMPLE_SCORE)

    assert result["technical_founder"]["score"] is None
    assert result["accelerator_partner"]["score"] is None
    assert result["final_verdict"]["label"] == "unknown"
    assert result["final_verdict"]["score"] is None


def test_debate_does_not_raise_when_no_api_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        try:
            debate_paper_with_two_agents(_SAMPLE_PAPER, _SAMPLE_SCORE)
        except Exception as exc:
            pytest.fail(f"debate_paper_with_two_agents raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# Mocked Anthropic call → JSON parsed correctly
# ---------------------------------------------------------------------------

def test_debate_parses_valid_claude_response() -> None:
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
         patch("anthropic_client.claude_chat", return_value=_VALID_VERDICT_JSON):
        result = debate_paper_with_two_agents(_SAMPLE_PAPER, _SAMPLE_SCORE)

    assert result["technical_founder"]["score"] == 4
    assert result["accelerator_partner"]["score"] == 5
    assert result["final_verdict"]["label"] == "keep"
    assert result["final_verdict"]["score"] == 4


def test_debate_returns_placeholder_when_claude_fails_both_attempts() -> None:
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
         patch("anthropic_client.claude_chat", side_effect=RuntimeError("API error")):
        result = debate_paper_with_two_agents(_SAMPLE_PAPER, _SAMPLE_SCORE)

    assert result["final_verdict"]["label"] == "unknown"


# ---------------------------------------------------------------------------
# _parse_verdict unit tests
# ---------------------------------------------------------------------------

def test_parse_verdict_valid_json() -> None:
    result = _parse_verdict(_VALID_VERDICT_JSON)
    assert result["final_verdict"]["label"] == "keep"


def test_parse_verdict_with_surrounding_prose() -> None:
    noisy = f"Here is the verdict:\n{_VALID_VERDICT_JSON}\nThanks!"
    result = _parse_verdict(noisy)
    assert result["technical_founder"]["score"] == 4


def test_parse_verdict_raises_on_missing_keys() -> None:
    bad_json = '{"technical_founder": {"score": 4, "rationale": "ok"}}'
    with pytest.raises(RuntimeError, match="missing required keys"):
        _parse_verdict(bad_json)
