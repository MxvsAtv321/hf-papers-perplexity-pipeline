from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from llm_client import _parse_analysis_json, score_paper_for_startup, validate_analysis_schema
from models import Paper

_SAMPLE_PAPER = Paper(
    paper_id="2501.00001",
    title="A New Agentic Framework",
    url="https://huggingface.co/papers/2501.00001",
    abstract="We propose a framework for agentic task planning.",
    published_at=datetime(2026, 2, 22, tzinfo=UTC),
)


def test_parse_analysis_json_with_wrapping_text() -> None:
    wrapped = (
        "Here is your result:\n"
        '{"summary":{"problem":"","core_method":"","key_technical_idea":"","inputs_outputs":"",'
        '"data_assumptions":"","metrics_and_baselines":"","limitations":""},'
        '"capability":{"plain_language_capability":""},"product_angles":[],'
        '"competition":[],"top_bets":[]}\n'
        "Thanks!"
    )

    parsed = _parse_analysis_json(wrapped)
    assert isinstance(parsed, dict)
    assert validate_analysis_schema(parsed) is True


def test_validate_analysis_schema_missing_top_key() -> None:
    data = {
        "summary": {
            "problem": "",
            "core_method": "",
            "key_technical_idea": "",
            "inputs_outputs": "",
            "data_assumptions": "",
            "metrics_and_baselines": "",
            "limitations": "",
        },
        "capability": {"plain_language_capability": ""},
        "product_angles": [],
        # "competition" is missing
        "top_bets": [],
    }
    assert validate_analysis_schema(data) is False


def test_score_paper_for_startup_returns_all_keys() -> None:
    """score_paper_for_startup parses a mocked OpenAI response into a dict with all keys."""
    mock_content = (
        '{"startup_potential": 4, "market_pull": 3, "technical_moat": 5, '
        '"story_for_accelerator": 4, "overall_score": 4, "rationale": "Strong technical moat."}'
    )
    mock_choice = MagicMock()
    mock_choice.message.content = mock_content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("llm_client.OpenAI", return_value=mock_client), \
         patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        result = score_paper_for_startup(_SAMPLE_PAPER)

    assert result["startup_potential"] == 4
    assert result["market_pull"] == 3
    assert result["technical_moat"] == 5
    assert result["overall_score"] == 4
    assert "rationale" in result


def test_score_paper_for_startup_raises_without_api_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            score_paper_for_startup(_SAMPLE_PAPER)


def test_validate_analysis_schema_missing_summary_field() -> None:
    data = {
        "summary": {
            "problem": "",
            # "core_method" is missing
            "key_technical_idea": "",
            "inputs_outputs": "",
            "data_assumptions": "",
            "metrics_and_baselines": "",
            "limitations": "",
        },
        "capability": {"plain_language_capability": ""},
        "product_angles": [],
        "competition": [],
        "top_bets": [],
    }
    assert validate_analysis_schema(data) is False
