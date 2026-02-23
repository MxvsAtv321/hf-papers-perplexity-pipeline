"""Tests for claude_agentic.debate_paper_with_two_agents and normalize_claude_verdict."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from claude_agentic import _parse_verdict, debate_paper_with_two_agents, normalize_claude_verdict
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

# Response from Call 1 (Technical Founder)
_TF_RESPONSE_JSON = '{"score": 4, "rationale": "Novel architecture."}'

# Response from Call 2 (Accelerator Partner)
_AP_RESPONSE_JSON = """{
  "accelerator_partner": {"score": 5, "rationale": "Large TAM."},
  "final_verdict": {"score": 4, "label": "keep", "reason": "Worth investing."}
}"""

# Full valid verdict (used by _parse_verdict tests)
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
# Mocked Anthropic calls → two sequential calls, JSON parsed correctly
# ---------------------------------------------------------------------------

def test_debate_makes_two_sequential_api_calls() -> None:
    """Each agent must trigger an independent claude_chat call."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
         patch("anthropic_client.claude_chat",
               side_effect=[_TF_RESPONSE_JSON, _AP_RESPONSE_JSON]) as mock_chat:
        debate_paper_with_two_agents(_SAMPLE_PAPER, _SAMPLE_SCORE)

    assert mock_chat.call_count == 2


def test_debate_parses_valid_claude_response() -> None:
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
         patch("anthropic_client.claude_chat",
               side_effect=[_TF_RESPONSE_JSON, _AP_RESPONSE_JSON]):
        result = debate_paper_with_two_agents(_SAMPLE_PAPER, _SAMPLE_SCORE)

    assert result["technical_founder"]["score"] == 4
    assert result["technical_founder"]["rationale"] == "Novel architecture."
    assert result["accelerator_partner"]["score"] == 5
    assert result["final_verdict"]["label"] == "keep"
    assert result["final_verdict"]["score"] == 4


def test_debate_tf_view_passed_to_ap_agent() -> None:
    """AP agent's user message must contain the TF's JSON output."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
         patch("anthropic_client.claude_chat",
               side_effect=[_TF_RESPONSE_JSON, _AP_RESPONSE_JSON]) as mock_chat:
        debate_paper_with_two_agents(_SAMPLE_PAPER, _SAMPLE_SCORE)

    ap_call_messages = mock_chat.call_args_list[1][0][0]  # positional args[0] = messages
    ap_user_content = next(m["content"] for m in ap_call_messages if m["role"] == "user")
    assert "Technical Founder" in ap_user_content
    assert "Novel architecture." in ap_user_content


def test_debate_returns_placeholder_when_claude_fails_both_attempts() -> None:
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
         patch("anthropic_client.claude_chat", side_effect=RuntimeError("API error")):
        result = debate_paper_with_two_agents(_SAMPLE_PAPER, _SAMPLE_SCORE)

    assert result["final_verdict"]["label"] == "unknown"


def test_debate_retries_on_failure_then_succeeds() -> None:
    """If attempt 1 fails partway through, attempt 2 should succeed."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
         patch("anthropic_client.claude_chat",
               side_effect=[RuntimeError("transient"), _TF_RESPONSE_JSON, _AP_RESPONSE_JSON]):
        result = debate_paper_with_two_agents(_SAMPLE_PAPER, _SAMPLE_SCORE)

    assert result["final_verdict"]["label"] == "keep"


def test_debate_result_is_always_normalized() -> None:
    """debate_paper_with_two_agents must always return all required keys."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
         patch("anthropic_client.claude_chat",
               side_effect=[_TF_RESPONSE_JSON, _AP_RESPONSE_JSON]):
        result = debate_paper_with_two_agents(_SAMPLE_PAPER, _SAMPLE_SCORE)

    # All top-level keys must be present
    assert "technical_founder" in result
    assert "accelerator_partner" in result
    assert "final_verdict" in result
    # All nested keys must be present
    assert "score" in result["technical_founder"]
    assert "rationale" in result["technical_founder"]
    assert "score" in result["accelerator_partner"]
    assert "rationale" in result["accelerator_partner"]
    assert "score" in result["final_verdict"]
    assert "label" in result["final_verdict"]
    assert "reason" in result["final_verdict"]


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


# ---------------------------------------------------------------------------
# normalize_claude_verdict unit tests
# ---------------------------------------------------------------------------

def test_normalize_verdict_passthrough_on_valid_dict() -> None:
    raw = {
        "technical_founder": {"score": 4, "rationale": "Strong moat."},
        "accelerator_partner": {"score": 5, "rationale": "Large market."},
        "final_verdict": {"score": 4, "label": "keep", "reason": "Good pick."},
    }
    result = normalize_claude_verdict(raw)

    assert result["technical_founder"]["score"] == 4
    assert result["accelerator_partner"]["score"] == 5
    assert result["final_verdict"]["label"] == "keep"


def test_normalize_verdict_fills_missing_keys() -> None:
    """A completely empty dict should produce a fully-structured verdict."""
    result = normalize_claude_verdict({})

    assert result["technical_founder"]["score"] is None
    assert result["technical_founder"]["rationale"] == ""
    assert result["accelerator_partner"]["score"] is None
    assert result["accelerator_partner"]["rationale"] == ""
    assert result["final_verdict"]["score"] is None
    assert result["final_verdict"]["label"] == "unknown"
    assert result["final_verdict"]["reason"] == ""


def test_normalize_verdict_coerces_string_score_to_int() -> None:
    raw = {
        "technical_founder": {"score": "3", "rationale": "ok"},
        "accelerator_partner": {"score": "4", "rationale": "ok"},
        "final_verdict": {"score": "3", "label": "maybe", "reason": "ok"},
    }
    result = normalize_claude_verdict(raw)

    assert result["technical_founder"]["score"] == 3
    assert isinstance(result["technical_founder"]["score"], int)


def test_normalize_verdict_returns_none_for_unparseable_score() -> None:
    raw = {
        "technical_founder": {"score": "strong", "rationale": "ok"},
        "accelerator_partner": {"score": None, "rationale": "ok"},
        "final_verdict": {"score": "N/A", "label": "drop", "reason": "ok"},
    }
    result = normalize_claude_verdict(raw)

    assert result["technical_founder"]["score"] is None
    assert result["accelerator_partner"]["score"] is None
    assert result["final_verdict"]["score"] is None


def test_normalize_verdict_truncates_long_rationale() -> None:
    long_text = "A" * 600
    raw = {
        "technical_founder": {"score": 4, "rationale": long_text},
        "accelerator_partner": {"score": 4, "rationale": "ok"},
        "final_verdict": {"score": 4, "label": "keep", "reason": long_text},
    }
    result = normalize_claude_verdict(raw)

    assert len(result["technical_founder"]["rationale"]) <= 500
    assert len(result["final_verdict"]["reason"]) <= 500


def test_normalize_verdict_handles_placeholder() -> None:
    """The placeholder dict should normalize cleanly with None scores preserved."""
    placeholder = {
        "technical_founder": {"score": None, "rationale": "claude_not_configured"},
        "accelerator_partner": {"score": None, "rationale": "claude_not_configured"},
        "final_verdict": {"score": None, "label": "unknown", "reason": "Claude API not configured"},
    }
    result = normalize_claude_verdict(placeholder)

    assert result["technical_founder"]["score"] is None
    assert result["final_verdict"]["label"] == "unknown"
    assert result["technical_founder"]["rationale"] == "claude_not_configured"


def test_normalize_verdict_missing_nested_keys() -> None:
    """Partial dicts (missing score or rationale) should be filled with defaults."""
    raw = {
        "technical_founder": {"score": 3},       # missing rationale
        "accelerator_partner": {"rationale": "ok"},  # missing score
        "final_verdict": {"label": "keep"},       # missing score and reason
    }
    result = normalize_claude_verdict(raw)

    assert result["technical_founder"]["rationale"] == ""
    assert result["accelerator_partner"]["score"] is None
    assert result["final_verdict"]["score"] is None
    assert result["final_verdict"]["reason"] == ""
