"""Tests for the wide_scout pipeline mode (main.run_wide_scout)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

import main
from models import Paper


def _paper(paper_id: str) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        url=f"https://huggingface.co/papers/{paper_id}",
        abstract="abstract",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _score(overall: int = 4, startup: int = 4, market: int = 4) -> dict:
    return {
        "startup_potential": startup,
        "market_pull": market,
        "technical_moat": 3,
        "story_for_accelerator": 3,
        "overall_score": overall,
        "rationale": "ok",
    }


def test_wide_scout_caps_stage2_at_max_llm_papers() -> None:
    """Only max_llm_papers papers are scored even if more pass the filter."""
    papers = [_paper(f"000{i}") for i in range(5)]

    with patch("main.fetch_papers", return_value=papers), \
         patch("main.is_potential_startup_paper", return_value=True), \
         patch("main.paper_already_exists", return_value=False), \
         patch("main.score_paper_for_startup", return_value=_score()) as mock_score, \
         patch("main.write_paper_entry"), \
         patch("main.debate_paper_with_two_agents"), \
         patch("main.analyze_paper"):
        main.run_wide_scout(max_fetch_days=7, max_llm_papers=3, max_deep_papers=0, dry_run=False)

    assert mock_score.call_count == 3


def test_wide_scout_skips_stage3_and_4_when_max_deep_papers_is_zero() -> None:
    """With max_deep_papers=0, debate and analyze are never called."""
    papers = [_paper("0001")]

    with patch("main.fetch_papers", return_value=papers), \
         patch("main.is_potential_startup_paper", return_value=True), \
         patch("main.paper_already_exists", return_value=False), \
         patch("main.score_paper_for_startup", return_value=_score()), \
         patch("main.write_paper_entry"), \
         patch("main.debate_paper_with_two_agents") as mock_debate, \
         patch("main.analyze_paper") as mock_analyze:
        main.run_wide_scout(max_fetch_days=7, max_llm_papers=10, max_deep_papers=0, dry_run=False)

    mock_debate.assert_not_called()
    mock_analyze.assert_not_called()


def test_wide_scout_runs_stage3_4_for_top_n_papers() -> None:
    """When max_deep_papers=2, only the top-2-scoring papers get Stage 3-4."""
    papers = [_paper("0001"), _paper("0002"), _paper("0003")]
    scores = {
        "0001": _score(overall=5),
        "0002": _score(overall=2),  # lowest — should be excluded from deep
        "0003": _score(overall=4),
    }

    with patch("main.fetch_papers", return_value=papers), \
         patch("main.is_potential_startup_paper", return_value=True), \
         patch("main.paper_already_exists", return_value=False), \
         patch("main.score_paper_for_startup", side_effect=lambda p: scores[p.paper_id]), \
         patch("main.write_paper_entry"), \
         patch("main.debate_paper_with_two_agents", return_value={}), \
         patch("main.analyze_paper", return_value={}) as mock_analyze:
        main.run_wide_scout(max_fetch_days=7, max_llm_papers=10, max_deep_papers=2, dry_run=False)

    # Top 2 by overall_score: 0001 (5) and 0003 (4) — 0002 (2) stays shallow
    assert mock_analyze.call_count == 2
    analyzed_ids = {c.args[0].paper_id for c in mock_analyze.call_args_list}
    assert "0001" in analyzed_ids
    assert "0003" in analyzed_ids
    assert "0002" not in analyzed_ids


def test_wide_scout_writes_to_wide_scout_csv_path() -> None:
    """write_paper_entry is called with csv_path=WIDE_SCOUT_CSV_PATH."""
    papers = [_paper("0001")]

    with patch("main.fetch_papers", return_value=papers), \
         patch("main.is_potential_startup_paper", return_value=True), \
         patch("main.paper_already_exists", return_value=False), \
         patch("main.score_paper_for_startup", return_value=_score()), \
         patch("main.write_paper_entry") as mock_write, \
         patch("main.debate_paper_with_two_agents"), \
         patch("main.analyze_paper"):
        main.run_wide_scout(max_fetch_days=7, max_llm_papers=10, max_deep_papers=0, dry_run=False)

    mock_write.assert_called_once()
    _, kwargs = mock_write.call_args
    assert kwargs.get("csv_path") == main.WIDE_SCOUT_CSV_PATH


def test_wide_scout_dedup_uses_wide_scout_csv_path() -> None:
    """paper_already_exists is called with csv_path=WIDE_SCOUT_CSV_PATH."""
    papers = [_paper("0001")]

    with patch("main.fetch_papers", return_value=papers), \
         patch("main.is_potential_startup_paper", return_value=True), \
         patch("main.paper_already_exists", return_value=False) as mock_exists, \
         patch("main.score_paper_for_startup", return_value=_score()), \
         patch("main.write_paper_entry"), \
         patch("main.debate_paper_with_two_agents"), \
         patch("main.analyze_paper"):
        main.run_wide_scout(max_fetch_days=7, max_llm_papers=10, max_deep_papers=0, dry_run=False)

    mock_exists.assert_called_once()
    _, kwargs = mock_exists.call_args
    assert kwargs.get("csv_path") == main.WIDE_SCOUT_CSV_PATH


def test_wide_scout_existing_papers_are_skipped() -> None:
    """Papers already in the wide scout CSV are not scored."""
    papers = [_paper("0001"), _paper("0002")]

    # 0001 already exists; 0002 is new
    def exists_side_effect(paper, csv_path=None):  # noqa: ARG001
        return paper.paper_id == "0001"

    with patch("main.fetch_papers", return_value=papers), \
         patch("main.is_potential_startup_paper", return_value=True), \
         patch("main.paper_already_exists", side_effect=exists_side_effect), \
         patch("main.score_paper_for_startup", return_value=_score()) as mock_score, \
         patch("main.write_paper_entry"):
        main.run_wide_scout(max_fetch_days=7, max_llm_papers=10, max_deep_papers=0, dry_run=False)

    assert mock_score.call_count == 1
    assert mock_score.call_args.args[0].paper_id == "0002"


def test_wide_scout_dry_run_no_api_calls_or_writes() -> None:
    """In dry-run mode, no API calls are made and nothing is written to CSV."""
    papers = [_paper("0001"), _paper("0002")]

    with patch("main.fetch_papers", return_value=papers), \
         patch("main.is_potential_startup_paper", return_value=True), \
         patch("main.paper_already_exists", return_value=False), \
         patch("main.score_paper_for_startup") as mock_score, \
         patch("main.write_paper_entry") as mock_write, \
         patch("main.debate_paper_with_two_agents") as mock_debate, \
         patch("main.analyze_paper") as mock_analyze:
        main.run_wide_scout(max_fetch_days=7, max_llm_papers=10, max_deep_papers=0, dry_run=True)

    mock_score.assert_not_called()
    mock_debate.assert_not_called()
    mock_analyze.assert_not_called()
    mock_write.assert_not_called()


def test_wide_scout_passes_fetch_days_to_fetch_papers() -> None:
    """run_wide_scout passes max_fetch_days as fetch_days= to fetch_papers."""
    with patch("main.fetch_papers", return_value=[]) as mock_fetch:
        main.run_wide_scout(max_fetch_days=45, max_llm_papers=10, max_deep_papers=0, dry_run=False)

    mock_fetch.assert_called_once_with(fetch_days=45)


def test_wide_scout_stage2_only_rows_have_stage3_and_4_false() -> None:
    """Papers not in the deep set are written with stage3_debated=stage4_analyzed=False."""
    papers = [_paper("0001")]

    written_stages: dict = {}

    def capture_write(paper, analysis, score=None, debate=None, stages=None, csv_path=None):  # noqa: ARG001
        written_stages.update(stages or {})

    with patch("main.fetch_papers", return_value=papers), \
         patch("main.is_potential_startup_paper", return_value=True), \
         patch("main.paper_already_exists", return_value=False), \
         patch("main.score_paper_for_startup", return_value=_score()), \
         patch("main.write_paper_entry", side_effect=capture_write), \
         patch("main.debate_paper_with_two_agents"), \
         patch("main.analyze_paper"):
        main.run_wide_scout(max_fetch_days=7, max_llm_papers=10, max_deep_papers=0, dry_run=False)

    assert written_stages["stage1_passed"] is True
    assert written_stages["stage2_scored"] is True
    assert written_stages["stage3_debated"] is False
    assert written_stages["stage4_analyzed"] is False
