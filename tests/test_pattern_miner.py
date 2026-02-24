"""Unit tests for pattern_miner.py."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from pattern_miner import (
    AnalyzedPaper,
    CompositeIdea,
    ScoredCompositeIdea,
    Theme,
    COMPOSITES_CSV_COLUMNS,
    CROSS_THEME_CSV_COLUMNS,
    CROSS_THEME_INTERSECTIONS,
    SCORED_COMPOSITES_CSV_COLUMNS,
    SCORED_WITH_SUMMARY_CSV_COLUMNS,
    THEMES_CSV_COLUMNS,
    _are_complementary,
    _collapse_product_angles,
    _row_to_paper,
    _select_cross_theme_candidates,
    add_simple_summaries_to_composites,
    aggregate_paper_signals,
    build_composite_prompt,
    build_cross_theme_prompt,
    build_scoring_prompt,
    build_simple_summary_prompt,
    build_theme_summary_prompt,
    composite_ideas_to_rows,
    extract_themes,
    find_composite_candidates,
    generate_cross_theme_composites,
    load_composite_ideas,
    load_papers,
    load_themes_from_csv,
    parse_composite_response,
    parse_cross_theme_response,
    parse_score_response,
    parse_simple_summary_response,
    score_composite_ideas,
    score_paper_against_theme,
    scored_ideas_to_rows,
    themes_to_rows,
    write_csv,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _paper(
    paper_id: str = "2501.00001",
    title: str = "Test Paper",
    overall_score: int | None = 4,
    startup_potential: int | None = 4,
    market_pull: int | None = 3,
    technical_moat: int | None = 3,
    story_for_accelerator: int | None = 3,
    capability: str | None = None,
    product_angles_text: str | None = None,
    score_rationale: str | None = None,
    abstract: str = "",
    claude_final_score: int | None = None,
    claude_final_label: str | None = None,
) -> AnalyzedPaper:
    return AnalyzedPaper(
        paper_id=paper_id,
        title=title,
        url=f"https://huggingface.co/papers/{paper_id}",
        abstract=abstract,
        startup_potential=startup_potential,
        market_pull=market_pull,
        technical_moat=technical_moat,
        story_for_accelerator=story_for_accelerator,
        overall_score=overall_score,
        claude_final_score=claude_final_score,
        claude_final_label=claude_final_label,
        capability=capability,
        product_angles_text=product_angles_text,
        score_rationale=score_rationale,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _row_to_paper — both CSV schemas
# ─────────────────────────────────────────────────────────────────────────────

def test_row_to_paper_pipeline_schema() -> None:
    row = {
        "paper_id": "2501.12345",
        "title": "My Test Paper",
        "url": "https://huggingface.co/papers/2501.12345",
        "abstract": "An abstract.",
        "overall_score": "4",
        "startup_potential": "5",
        "market_pull": "3",
        "technical_moat": "4",
        "story_for_accelerator": "3",
        "claude_final_score": "4",
        "claude_final_label": "keep",
        "capability_plain_language_capability": "It does X very well.",
        "product_angles": json.dumps([{"name": "Angle A"}, {"name": "Angle B"}]),
        "score_rationale": "Strong market pull.",
    }
    paper = _row_to_paper(row)
    assert paper is not None
    assert paper.paper_id == "2501.12345"
    assert paper.overall_score == 4
    assert paper.startup_potential == 5
    assert paper.capability == "It does X very well."
    assert paper.product_angles_text == "Angle A, Angle B"
    assert paper.score_rationale == "Strong market pull."


def test_row_to_paper_report_schema() -> None:
    """papers_debated.csv uses different column names."""
    row = {
        "title": "Debated Paper",
        "url": "https://huggingface.co/papers/2502.99999",
        "openai_overall": "5",
        "openai_startup": "4",
        "openai_market": "5",
        "openai_moat": "3",
        "openai_story": "4",
        "final_score": "4",
        "final_label": "keep",
        "capability": "Capable of doing Y.",
        "best_product_angle": "Enterprise search SDK",
    }
    paper = _row_to_paper(row)
    assert paper is not None
    assert paper.paper_id == "2502.99999"  # extracted from URL
    assert paper.overall_score == 5
    assert paper.market_pull == 5
    assert paper.capability == "Capable of doing Y."
    assert paper.claude_final_label == "keep"


def test_row_to_paper_missing_title_returns_none() -> None:
    row = {"paper_id": "2501.00001", "overall_score": "4"}
    assert _row_to_paper(row) is None


def test_row_to_paper_missing_id_and_url_returns_none() -> None:
    row = {"title": "A Paper"}
    assert _row_to_paper(row) is None


def test_row_to_paper_extracts_paper_id_from_url() -> None:
    row = {
        "title": "URL-only Paper",
        "url": "https://huggingface.co/papers/2501.56789",
        "overall_score": "3",
    }
    paper = _row_to_paper(row)
    assert paper is not None
    assert paper.paper_id == "2501.56789"


# ─────────────────────────────────────────────────────────────────────────────
# _collapse_product_angles
# ─────────────────────────────────────────────────────────────────────────────

def test_collapse_product_angles_json_array() -> None:
    raw = json.dumps([{"name": "Angle A"}, {"name": "Angle B"}])
    result = _collapse_product_angles(raw)
    assert result == "Angle A, Angle B"


def test_collapse_product_angles_empty_array() -> None:
    assert _collapse_product_angles("[]") is None


def test_collapse_product_angles_plain_string() -> None:
    assert _collapse_product_angles("Enterprise SDK") == "Enterprise SDK"


def test_collapse_product_angles_empty_string() -> None:
    assert _collapse_product_angles("") is None


# ─────────────────────────────────────────────────────────────────────────────
# load_papers
# ─────────────────────────────────────────────────────────────────────────────

def _write_test_csv(path: Path, rows: list[dict]) -> None:
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def test_load_papers_basic(tmp_path: Path) -> None:
    csv_path = tmp_path / "test.csv"
    _write_test_csv(csv_path, [
        {"paper_id": "0001", "title": "Paper A", "url": "https://huggingface.co/papers/0001",
         "overall_score": "4", "startup_potential": "4", "market_pull": "3",
         "technical_moat": "3", "story_for_accelerator": "3",
         "abstract": "", "score_rationale": "", "capability_plain_language_capability": "",
         "product_angles": "[]", "claude_final_score": "", "claude_final_label": ""},
    ])
    papers = load_papers([str(csv_path)], min_score=1)
    assert len(papers) == 1
    assert papers[0].paper_id == "0001"


def test_load_papers_filters_by_min_score(tmp_path: Path) -> None:
    csv_path = tmp_path / "test.csv"
    _write_test_csv(csv_path, [
        {"paper_id": "0001", "title": "High", "url": "https://huggingface.co/papers/0001",
         "overall_score": "4", "startup_potential": "4", "market_pull": "4",
         "technical_moat": "4", "story_for_accelerator": "4",
         "abstract": "", "score_rationale": "", "capability_plain_language_capability": "",
         "product_angles": "[]", "claude_final_score": "", "claude_final_label": ""},
        {"paper_id": "0002", "title": "Low", "url": "https://huggingface.co/papers/0002",
         "overall_score": "2", "startup_potential": "2", "market_pull": "2",
         "technical_moat": "2", "story_for_accelerator": "2",
         "abstract": "", "score_rationale": "", "capability_plain_language_capability": "",
         "product_angles": "[]", "claude_final_score": "", "claude_final_label": ""},
    ])
    papers = load_papers([str(csv_path)], min_score=3)
    assert len(papers) == 1
    assert papers[0].paper_id == "0001"


def test_load_papers_deduplicates_across_files(tmp_path: Path) -> None:
    row = {
        "paper_id": "0001", "title": "Dup Paper", "url": "https://huggingface.co/papers/0001",
        "overall_score": "4", "startup_potential": "4", "market_pull": "3",
        "technical_moat": "3", "story_for_accelerator": "3",
        "abstract": "", "score_rationale": "", "capability_plain_language_capability": "",
        "product_angles": "[]", "claude_final_score": "", "claude_final_label": "",
    }
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    _write_test_csv(p1, [row])
    _write_test_csv(p2, [row])
    papers = load_papers([str(p1), str(p2)], min_score=1)
    assert len(papers) == 1


def test_load_papers_skips_missing_file(tmp_path: Path) -> None:
    papers = load_papers([str(tmp_path / "nonexistent.csv")], min_score=1)
    assert papers == []


# ─────────────────────────────────────────────────────────────────────────────
# score_paper_against_theme
# ─────────────────────────────────────────────────────────────────────────────

def test_score_paper_against_theme_matches_title_keyword() -> None:
    paper = _paper(title="A robot manipulation system")
    score = score_paper_against_theme(paper, ["robot", "manipulation"])
    assert score == 2


def test_score_paper_against_theme_zero_when_no_match() -> None:
    paper = _paper(title="Quantum computing at scale")
    score = score_paper_against_theme(paper, ["robot", "manipulation"])
    assert score == 0


def test_score_paper_against_theme_uses_capability() -> None:
    paper = _paper(capability="Streams video frames and builds 3D gaussian splat scenes")
    score = score_paper_against_theme(paper, ["gaussian splat", "3d"])
    assert score == 2


# ─────────────────────────────────────────────────────────────────────────────
# extract_themes
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_themes_assigns_paper_to_best_theme() -> None:
    papers = [
        _paper("0001", title="Robot manipulation with sim-to-real transfer", abstract="robotics embodied"),
        _paper("0002", title="Web agent for browser automation", abstract="web agent rpa"),
    ]
    themes = extract_themes(papers)
    theme_map = {t.name: t.paper_ids for t in themes}

    # Each paper should be in exactly one non-Uncategorized theme
    robotics_theme = theme_map.get("Robotics & Embodied AI", [])
    web_theme = theme_map.get("Web & Computer-Use Agents", [])
    assert "0001" in robotics_theme
    assert "0002" in web_theme


def test_extract_themes_unmatched_goes_to_uncategorized() -> None:
    paper = _paper("9999", title="Quantum crystallography algorithms")
    themes = extract_themes([paper])
    theme_map = {t.name: t.paper_ids for t in themes}
    assert "9999" in theme_map.get("Uncategorized", [])


def test_extract_themes_each_paper_in_exactly_one_theme() -> None:
    papers = [_paper(f"000{i}", title=f"Paper {i}") for i in range(5)]
    themes = extract_themes(papers)
    all_assigned = [pid for t in themes for pid in t.paper_ids]
    # No duplicates
    assert len(all_assigned) == len(set(all_assigned)) == 5


def test_extract_themes_no_empty_themes() -> None:
    papers = [_paper("0001", title="Web agent browsing automation")]
    themes = extract_themes(papers)
    for t in themes:
        assert len(t.paper_ids) > 0


def test_extract_themes_sorted_by_size() -> None:
    papers = [_paper(f"0{i}", title="Robot manipulation embodied") for i in range(4)]
    papers += [_paper("99", title="Web agent browsing")]
    themes = extract_themes(papers)
    # Filter out Uncategorized for sorting check
    non_uncategorized = [t for t in themes if t.name != "Uncategorized"]
    sizes = [len(t.paper_ids) for t in non_uncategorized]
    assert sizes == sorted(sizes, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# _are_complementary + find_composite_candidates
# ─────────────────────────────────────────────────────────────────────────────

def test_are_complementary_tech_moat_plus_market_pull() -> None:
    p1 = _paper("0001", technical_moat=4, market_pull=2)
    p2 = _paper("0002", technical_moat=2, market_pull=4)
    assert _are_complementary(p1, p2) is True


def test_are_complementary_false_when_same_strengths() -> None:
    p1 = _paper("0001", technical_moat=4, market_pull=4)
    p2 = _paper("0002", technical_moat=4, market_pull=4)
    # Both strong in everything — still qualifies (moat>=4 and pull>=4 cross-check passes)
    # Actually this should be True too since moat1>=4 and pull2>=4
    assert _are_complementary(p1, p2) is True


def test_are_complementary_deep_plus_shallow_high_startup() -> None:
    p_deep = _paper("0001", capability="It does 3D reconstruction in real time")
    p_shallow = _paper("0002", capability=None, startup_potential=4)
    assert _are_complementary(p_deep, p_shallow) is True


def test_are_complementary_false_both_shallow_weak() -> None:
    p1 = _paper("0001", technical_moat=2, market_pull=2, story_for_accelerator=2, capability=None, startup_potential=2)
    p2 = _paper("0002", technical_moat=2, market_pull=2, story_for_accelerator=2, capability=None, startup_potential=2)
    assert _are_complementary(p1, p2) is False


def test_find_composite_candidates_returns_complementary_pairs() -> None:
    p1 = _paper("0001", title="Deep tech moat", technical_moat=4, market_pull=2, overall_score=4)
    p2 = _paper("0002", title="Market pull paper", technical_moat=2, market_pull=4, overall_score=4)
    p3 = _paper("0003", title="Mediocre paper", technical_moat=2, market_pull=2, overall_score=3)

    theme = Theme(name="Test Theme", keywords=[], paper_ids=["0001", "0002", "0003"], summary=None)
    papers_by_id = {p.paper_id: p for p in [p1, p2, p3]}

    candidates = find_composite_candidates(theme, papers_by_id)
    assert len(candidates) >= 1
    assert any(
        {p.paper_id for p in pair} == {"0001", "0002"}
        for pair in candidates
    )


def test_find_composite_candidates_excludes_low_score_papers() -> None:
    p1 = _paper("0001", technical_moat=4, market_pull=2, overall_score=4)
    p2 = _paper("0002", technical_moat=2, market_pull=4, overall_score=2)  # below min
    theme = Theme(name="T", keywords=[], paper_ids=["0001", "0002"], summary=None)
    papers_by_id = {"0001": p1, "0002": p2}
    candidates = find_composite_candidates(theme, papers_by_id)
    assert candidates == []


def test_find_composite_candidates_max_three_per_theme() -> None:
    # Create 10 eligible complementary pairs
    papers = [
        _paper(f"000{i}", technical_moat=4 if i % 2 == 0 else 2,
               market_pull=2 if i % 2 == 0 else 4, overall_score=4)
        for i in range(10)
    ]
    theme = Theme(name="T", keywords=[], paper_ids=[p.paper_id for p in papers], summary=None)
    papers_by_id = {p.paper_id: p for p in papers}
    candidates = find_composite_candidates(theme, papers_by_id)
    assert len(candidates) <= 3


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def test_build_theme_summary_prompt_structure() -> None:
    theme = Theme(name="Robotics", keywords=["robot"], paper_ids=["0001"], summary=None)
    papers = [_paper("0001", title="Robot arm control", capability="Controls robot arms.")]
    messages = build_theme_summary_prompt(theme, papers)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Robotics" in messages[1]["content"]
    assert "Robot arm control" in messages[1]["content"]


def test_build_composite_prompt_structure() -> None:
    theme = Theme(name="Web Agents", keywords=["web"], paper_ids=["0001", "0002"], summary=None)
    candidates = [
        _paper("0001", title="WebSim", capability="Simulates web interactions."),
        _paper("0002", title="WebRouter", capability="Routes web agent tasks."),
    ]
    messages = build_composite_prompt(theme, candidates)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    user_content = messages[1]["content"]
    assert "WebSim" in user_content
    assert "WebRouter" in user_content
    assert "Web Agents" in user_content


def test_build_composite_prompt_includes_scores() -> None:
    theme = Theme(name="T", keywords=[], paper_ids=[], summary=None)
    candidates = [_paper("0001", overall_score=5, market_pull=4)]
    messages = build_composite_prompt(theme, candidates)
    assert "overall=5" in messages[1]["content"]
    assert "market=4" in messages[1]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# parse_composite_response
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_composite_response_keep() -> None:
    raw = json.dumps({
        "combine": True,
        "title": "Agent Training Sim",
        "core_capability": "Trains agents offline at scale.",
        "added_value": "Faster iteration without real-world risk.",
        "target_user": "Robotics engineer",
        "problem_statement": "Collecting real trajectories is slow.",
        "wedge_description": "Sell as a data engine SaaS.",
        "risks": "Sim-to-real gap remains a challenge.",
    })
    idea = parse_composite_response(raw, "Robotics", ["0001", "0002"])
    assert idea is not None
    assert idea.title == "Agent Training Sim"
    assert idea.theme_name == "Robotics"
    assert idea.paper_ids == ["0001", "0002"]
    assert len(idea.id) > 0


def test_parse_composite_response_no_combine() -> None:
    raw = json.dumps({"combine": False, "reason": "Papers too similar."})
    idea = parse_composite_response(raw, "Robotics", ["0001", "0002"])
    assert idea is None


def test_parse_composite_response_invalid_json_returns_none() -> None:
    idea = parse_composite_response("not json at all ~~~", "T", ["0001"])
    assert idea is None


# ─────────────────────────────────────────────────────────────────────────────
# CSV serialization helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_themes_to_rows_schema() -> None:
    theme = Theme(name="Web Agents", keywords=["web", "agent"], paper_ids=["0001", "0002"], summary="Summary text.")
    rows = themes_to_rows([theme])
    assert len(rows) == 1
    row = rows[0]
    assert row["theme_name"] == "Web Agents"
    assert row["num_papers"] == 2
    assert row["summary"] == "Summary text."
    assert json.loads(row["paper_ids"]) == ["0001", "0002"]


def test_themes_to_rows_no_summary() -> None:
    theme = Theme(name="T", keywords=[], paper_ids=[], summary=None)
    rows = themes_to_rows([theme])
    assert rows[0]["summary"] == ""


def test_composite_ideas_to_rows_schema() -> None:
    idea = CompositeIdea(
        id="abc12345",
        title="Sim + Router",
        theme_name="Web Agents",
        paper_ids=["0001", "0002"],
        core_capability="Simulates and routes.",
        added_value="Better together.",
        target_user="ML engineer",
        problem_statement="Hard to train agents.",
        wedge_description="Narrow SaaS entry.",
        risks="Competitive moat weak.",
    )
    rows = composite_ideas_to_rows([idea])
    assert len(rows) == 1
    row = rows[0]
    assert row["title"] == "Sim + Router"
    assert json.loads(row["paper_ids"]) == ["0001", "0002"]
    assert row["wedge_description"] == "Narrow SaaS entry."


def test_write_csv_creates_file(tmp_path: Path) -> None:
    path = str(tmp_path / "out.csv")
    rows = [{"theme_name": "T", "num_papers": 1, "keywords": "kw", "paper_ids": "[]", "summary": ""}]
    write_csv(path, THEMES_CSV_COLUMNS, rows)
    assert os.path.exists(path)
    with open(path, newline="", encoding="utf-8") as fh:
        content = list(csv.DictReader(fh))
    assert len(content) == 1
    assert content[0]["theme_name"] == "T"


# ─────────────────────────────────────────────────────────────────────────────
# load_composite_ideas
# ─────────────────────────────────────────────────────────────────────────────

def _write_composite_ideas_csv(path: Path, ideas: list[CompositeIdea]) -> None:
    rows = composite_ideas_to_rows(ideas)
    write_csv(str(path), COMPOSITES_CSV_COLUMNS, rows)


def test_load_composite_ideas_basic(tmp_path: Path) -> None:
    idea = CompositeIdea(
        id="abc12345",
        title="Test Idea",
        theme_name="Robotics",
        paper_ids=["0001", "0002"],
        core_capability="Does something novel.",
        added_value="Better together.",
        target_user="ML engineer",
        problem_statement="Hard to deploy.",
        wedge_description="Narrow SaaS.",
        risks="Hard to scale.",
    )
    csv_path = tmp_path / "ideas.csv"
    _write_composite_ideas_csv(csv_path, [idea])
    loaded = load_composite_ideas(str(csv_path))
    assert len(loaded) == 1
    assert loaded[0].title == "Test Idea"
    assert loaded[0].paper_ids == ["0001", "0002"]
    assert loaded[0].theme_name == "Robotics"


def test_load_composite_ideas_skips_missing_file(tmp_path: Path) -> None:
    loaded = load_composite_ideas(str(tmp_path / "nonexistent.csv"))
    assert loaded == []


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_paper_signals
# ─────────────────────────────────────────────────────────────────────────────

def test_aggregate_paper_signals_basic() -> None:
    p1 = _paper("0001", overall_score=4, market_pull=3, technical_moat=4)
    p2 = _paper("0002", overall_score=5, market_pull=5, technical_moat=3)
    papers_by_id = {"0001": p1, "0002": p2}
    signals = aggregate_paper_signals(["0001", "0002"], papers_by_id)
    assert signals["avg_overall"] == 4.5
    assert signals["avg_market_pull"] == 4.0
    assert signals["avg_technical_moat"] == 3.5
    assert signals["num_papers"] == 2


def test_aggregate_paper_signals_empty_ids() -> None:
    signals = aggregate_paper_signals([], {})
    assert signals["avg_overall"] is None
    assert signals["has_claude"] is False
    assert signals["num_papers"] == 0


def test_aggregate_paper_signals_has_claude_true() -> None:
    p = _paper("0001", claude_final_score=4)
    signals = aggregate_paper_signals(["0001"], {"0001": p})
    assert signals["has_claude"] is True
    assert signals["avg_claude_score"] == 4.0


def test_aggregate_paper_signals_has_claude_false_when_none() -> None:
    p = _paper("0001", claude_final_score=None)
    signals = aggregate_paper_signals(["0001"], {"0001": p})
    assert signals["has_claude"] is False
    assert signals["avg_claude_score"] is None


def test_aggregate_paper_signals_ignores_missing_ids() -> None:
    p = _paper("0001", overall_score=4)
    # "9999" not in papers_by_id — should be silently ignored
    signals = aggregate_paper_signals(["0001", "9999"], {"0001": p})
    assert signals["num_papers"] == 1
    assert signals["avg_overall"] == 4.0


# ─────────────────────────────────────────────────────────────────────────────
# build_scoring_prompt
# ─────────────────────────────────────────────────────────────────────────────

def _make_idea(title: str = "Test Idea") -> CompositeIdea:
    return CompositeIdea(
        id="abc12345",
        title=title,
        theme_name="Robotics",
        paper_ids=["0001", "0002"],
        core_capability="Does X.",
        added_value="Better.",
        target_user="Engineer",
        problem_statement="Hard.",
        wedge_description="Narrow SaaS.",
        risks="Competition.",
    )


def test_build_scoring_prompt_structure() -> None:
    idea = _make_idea()
    signals = {"avg_overall": 4.0, "avg_market_pull": 3.5, "avg_technical_moat": 4.0,
               "avg_startup_potential": 4.0, "avg_story": 3.0,
               "avg_claude_score": None, "has_claude": False, "num_papers": 2}
    messages = build_scoring_prompt(idea, signals)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Test Idea" in messages[1]["content"]
    assert "Robotics" in messages[1]["content"]


def test_build_scoring_prompt_includes_claude_note_when_available() -> None:
    idea = _make_idea()
    signals = {"avg_overall": 4.5, "avg_market_pull": 4.0, "avg_technical_moat": 4.0,
               "avg_startup_potential": 4.0, "avg_story": 4.0,
               "avg_claude_score": 4.5, "has_claude": True, "num_papers": 2}
    messages = build_scoring_prompt(idea, signals)
    assert "4.5" in messages[1]["content"]
    assert "Claude debate score" in messages[1]["content"]


def test_build_scoring_prompt_no_claude_note_when_missing() -> None:
    idea = _make_idea()
    signals = {"avg_overall": 3.0, "avg_market_pull": 3.0, "avg_technical_moat": 3.0,
               "avg_startup_potential": 3.0, "avg_story": 3.0,
               "avg_claude_score": None, "has_claude": False, "num_papers": 2}
    messages = build_scoring_prompt(idea, signals)
    assert "Stage 2 signals only" in messages[1]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# parse_score_response
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_score_response_valid() -> None:
    raw = json.dumps({
        "wedge_clarity": 4,
        "technical_moat": 3,
        "market_pull": 5,
        "founder_fit": 4,
        "composite_synergy": 3,
        "scoring_notes": "Strong market pull but moat is weak.",
    })
    result = parse_score_response(raw)
    assert result is not None
    assert result["wedge_clarity"] == 4
    assert result["market_pull"] == 5
    assert result["scoring_notes"] == "Strong market pull but moat is weak."


def test_parse_score_response_computes_weighted_score() -> None:
    # All 5s → composite_score = 5.0
    raw = json.dumps({
        "wedge_clarity": 5, "technical_moat": 5, "market_pull": 5,
        "founder_fit": 5, "composite_synergy": 5, "scoring_notes": "Perfect.",
    })
    result = parse_score_response(raw)
    assert result is not None
    assert result["composite_score"] == 5.0


def test_parse_score_response_weighted_calculation() -> None:
    # wedge=4, moat=2, pull=4, fit=2, synergy=2
    # 0.25*4 + 0.20*2 + 0.25*4 + 0.15*2 + 0.15*2 = 1.0+0.4+1.0+0.3+0.3 = 3.0
    raw = json.dumps({
        "wedge_clarity": 4, "technical_moat": 2, "market_pull": 4,
        "founder_fit": 2, "composite_synergy": 2, "scoring_notes": "Mixed.",
    })
    result = parse_score_response(raw)
    assert result is not None
    assert result["composite_score"] == 3.0


def test_parse_score_response_invalid_json_returns_none() -> None:
    result = parse_score_response("not json at all ~~~")
    assert result is None


def test_parse_score_response_missing_fields_gives_none_composite() -> None:
    # Missing founder_fit → composite_score should be None
    raw = json.dumps({
        "wedge_clarity": 4, "technical_moat": 3, "market_pull": 4,
        "scoring_notes": "Incomplete.",
    })
    result = parse_score_response(raw)
    assert result is not None
    assert result["composite_score"] is None


# ─────────────────────────────────────────────────────────────────────────────
# score_composite_ideas
# ─────────────────────────────────────────────────────────────────────────────

def _good_llm_fn(messages: list[dict]) -> str:
    return json.dumps({
        "wedge_clarity": 4, "technical_moat": 3, "market_pull": 4,
        "founder_fit": 3, "composite_synergy": 4, "scoring_notes": "Looks good.",
    })


def _failing_llm_fn(messages: list[dict]) -> str:
    raise RuntimeError("API error")


def test_score_composite_ideas_success() -> None:
    idea = _make_idea()
    p = _paper("0001", overall_score=4)
    result = score_composite_ideas([idea], {"0001": p}, _good_llm_fn)
    assert len(result) == 1
    assert result[0].wedge_clarity == 4
    assert result[0].composite_score is not None
    assert result[0].scoring_notes == "Looks good."


def test_score_composite_ideas_fallback_on_failure() -> None:
    idea = _make_idea()
    result = score_composite_ideas([idea], {}, _failing_llm_fn)
    assert len(result) == 1
    assert result[0].composite_score is None
    assert result[0].scoring_notes == "[scoring failed]"


def test_score_composite_ideas_preserves_original_fields() -> None:
    idea = _make_idea("My Startup Idea")
    result = score_composite_ideas([idea], {}, _good_llm_fn)
    assert result[0].title == "My Startup Idea"
    assert result[0].theme_name == "Robotics"
    assert result[0].paper_ids == ["0001", "0002"]


# ─────────────────────────────────────────────────────────────────────────────
# scored_ideas_to_rows
# ─────────────────────────────────────────────────────────────────────────────

def _scored_idea(title: str, score: float | None) -> ScoredCompositeIdea:
    return ScoredCompositeIdea(
        id="abc12345", title=title, theme_name="T", paper_ids=["0001"],
        core_capability="X", added_value="Y", target_user="Z",
        problem_statement="P", wedge_description="W", risks="R",
        wedge_clarity=4 if score else None,
        technical_moat=3 if score else None,
        market_pull=4 if score else None,
        founder_fit=3 if score else None,
        composite_synergy=4 if score else None,
        composite_score=score,
        scoring_notes="Notes." if score else "[scoring failed]",
    )


def test_scored_ideas_to_rows_sorted_by_score_desc() -> None:
    ideas = [_scored_idea("Low", 2.5), _scored_idea("High", 4.5), _scored_idea("Mid", 3.5)]
    rows = scored_ideas_to_rows(ideas)
    scores = [float(r["composite_score"]) for r in rows if r["composite_score"] != ""]
    assert scores == sorted(scores, reverse=True)


def test_scored_ideas_to_rows_unscored_last() -> None:
    ideas = [_scored_idea("Scored", 3.5), _scored_idea("Failed", None)]
    rows = scored_ideas_to_rows(ideas)
    assert rows[0]["title"] == "Scored"
    assert rows[1]["title"] == "Failed"
    assert rows[1]["composite_score"] == ""


def test_scored_ideas_to_rows_schema() -> None:
    idea = _scored_idea("My Idea", 3.75)
    rows = scored_ideas_to_rows([idea])
    assert len(rows) == 1
    row = rows[0]
    for col in SCORED_COMPOSITES_CSV_COLUMNS:
        assert col in row
    assert row["title"] == "My Idea"
    assert row["composite_score"] == 3.75
    assert json.loads(row["paper_ids"]) == ["0001"]


# ─────────────────────────────────────────────────────────────────────────────
# build_simple_summary_prompt
# ─────────────────────────────────────────────────────────────────────────────

def _scored_row(title: str = "My Idea", score: str = "4.25") -> dict:
    return {
        "id": "abc12345",
        "title": title,
        "theme_name": "Robotics",
        "paper_ids": '["0001", "0002"]',
        "core_capability": "Trains robots to pick objects without manual labelling.",
        "added_value": "Combining two papers lets the system adapt to new objects faster.",
        "target_user": "Warehouse automation engineers at mid-size logistics companies.",
        "problem_statement": "Setting up a new robot gripper requires weeks of manual data collection.",
        "wedge_description": "Start with a narrow plug-in for one popular robot arm brand.",
        "risks": "Sim-to-real gap; large players may copy.",
        "wedge_clarity": "4", "technical_moat": "3", "market_pull": "4",
        "founder_fit": "3", "composite_synergy": "4",
        "composite_score": score,
        "scoring_notes": "Strong wedge, moderate moat.",
    }


def test_build_simple_summary_prompt_structure() -> None:
    row = _scored_row()
    messages = build_simple_summary_prompt(row)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_build_simple_summary_prompt_contains_key_fields() -> None:
    row = _scored_row("Robotic Gripping Suite", "4.25")
    messages = build_simple_summary_prompt(row)
    content = messages[1]["content"]
    assert "Robotic Gripping Suite" in content
    assert "Robotics" in content
    assert "4.25" in content


def test_build_simple_summary_prompt_no_score_graceful() -> None:
    row = _scored_row(score="")
    messages = build_simple_summary_prompt(row)
    assert "not scored" in messages[1]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# parse_simple_summary_response
# ─────────────────────────────────────────────────────────────────────────────

def _ok_summary() -> str:
    """A valid summary under 170 words."""
    return (
        "This product is for warehouse operations teams at mid-size logistics companies. "
        "It makes robot arms learn to pick up new objects without weeks of manual setup — "
        "you show it a few examples and it figures out the rest automatically. "
        "Most warehouse robots today need a specialist to reconfigure them every time a new "
        "product arrives, which causes expensive downtime. This tool cuts that setup time "
        "from weeks to hours by combining two research advances: one that generates training "
        "data automatically, and another that lets the robot adapt quickly on the job."
    )


def test_parse_simple_summary_response_valid() -> None:
    raw = json.dumps({"simple_summary": _ok_summary()})
    result = parse_simple_summary_response(raw)
    assert result is not None
    assert "warehouse" in result.lower()


def test_parse_simple_summary_response_too_long_returns_none() -> None:
    too_long = " ".join(["word"] * 200)
    raw = json.dumps({"simple_summary": too_long})
    result = parse_simple_summary_response(raw)
    assert result is None


def test_parse_simple_summary_response_empty_returns_none() -> None:
    raw = json.dumps({"simple_summary": ""})
    assert parse_simple_summary_response(raw) is None


def test_parse_simple_summary_response_invalid_json_returns_none() -> None:
    assert parse_simple_summary_response("not json ~~~") is None


def test_parse_simple_summary_response_strips_whitespace() -> None:
    raw = json.dumps({"simple_summary": "  Hello world.  "})
    result = parse_simple_summary_response(raw)
    assert result == "Hello world."


# ─────────────────────────────────────────────────────────────────────────────
# add_simple_summaries_to_composites
# ─────────────────────────────────────────────────────────────────────────────

def _write_scored_csv(path: Path, rows: list[dict]) -> None:
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def _good_summary_fn(messages: list[dict]) -> str:
    return json.dumps({"simple_summary": _ok_summary()})


def _failing_summary_fn(messages: list[dict]) -> str:
    raise RuntimeError("API error")


def test_add_simple_summaries_creates_output_file(tmp_path: Path) -> None:
    in_csv = tmp_path / "scored.csv"
    out_csv = tmp_path / "scored_with_summary.csv"
    _write_scored_csv(in_csv, [_scored_row("Idea A"), _scored_row("Idea B")])

    result = add_simple_summaries_to_composites(str(in_csv), str(out_csv), _good_summary_fn)

    assert os.path.exists(str(out_csv))
    assert len(result) == 2


def test_add_simple_summaries_output_has_simple_summary_column(tmp_path: Path) -> None:
    in_csv = tmp_path / "scored.csv"
    out_csv = tmp_path / "out.csv"
    _write_scored_csv(in_csv, [_scored_row("My Idea")])

    add_simple_summaries_to_composites(str(in_csv), str(out_csv), _good_summary_fn)

    with out_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert "simple_summary" in rows[0]
    assert rows[0]["simple_summary"] != ""
    assert rows[0]["title"] == "My Idea"


def test_add_simple_summaries_preserves_existing_columns(tmp_path: Path) -> None:
    in_csv = tmp_path / "scored.csv"
    out_csv = tmp_path / "out.csv"
    row = _scored_row("Preserve Me")
    _write_scored_csv(in_csv, [row])

    add_simple_summaries_to_composites(str(in_csv), str(out_csv), _good_summary_fn)

    with out_csv.open(newline="", encoding="utf-8") as fh:
        out_row = list(csv.DictReader(fh))[0]
    # All original columns still present
    for col in row.keys():
        assert col in out_row
    assert out_row["composite_score"] == row["composite_score"]


def test_add_simple_summaries_fallback_on_failure(tmp_path: Path) -> None:
    in_csv = tmp_path / "scored.csv"
    out_csv = tmp_path / "out.csv"
    _write_scored_csv(in_csv, [_scored_row("Failing Idea")])

    result = add_simple_summaries_to_composites(str(in_csv), str(out_csv), _failing_summary_fn)

    assert result[0]["simple_summary"] == "[summary unavailable]"


def test_add_simple_summaries_idempotent(tmp_path: Path) -> None:
    """Running twice overwrites output without duplicating rows."""
    in_csv = tmp_path / "scored.csv"
    out_csv = tmp_path / "out.csv"
    _write_scored_csv(in_csv, [_scored_row("Idea")])

    add_simple_summaries_to_composites(str(in_csv), str(out_csv), _good_summary_fn)
    add_simple_summaries_to_composites(str(in_csv), str(out_csv), _good_summary_fn)

    with out_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1  # not duplicated


def test_add_simple_summaries_skips_missing_input(tmp_path: Path) -> None:
    result = add_simple_summaries_to_composites(
        str(tmp_path / "nonexistent.csv"),
        str(tmp_path / "out.csv"),
        _good_summary_fn,
    )
    assert result == []


def test_scored_with_summary_csv_columns_includes_simple_summary() -> None:
    assert "simple_summary" in SCORED_WITH_SUMMARY_CSV_COLUMNS
    # All scored columns also present
    for col in SCORED_COMPOSITES_CSV_COLUMNS:
        assert col in SCORED_WITH_SUMMARY_CSV_COLUMNS


# ─────────────────────────────────────────────────────────────────────────────
# Cross-theme composite generation
# ─────────────────────────────────────────────────────────────────────────────

def _theme(name: str, paper_ids: list[str], summary: str = "") -> Theme:
    return Theme(name=name, keywords=[], paper_ids=paper_ids, summary=summary or None)


def test_select_cross_theme_candidates_prefers_keeps() -> None:
    keep = _paper("0001", overall_score=4, startup_potential=4, claude_final_label="keep")
    high = _paper("0002", overall_score=4, startup_potential=4, claude_final_label=None)
    low  = _paper("0003", overall_score=2, startup_potential=2, claude_final_label=None)
    theme = _theme("T", ["0001", "0002", "0003"])
    papers_by_id = {"0001": keep, "0002": high, "0003": low}

    result = _select_cross_theme_candidates(theme, papers_by_id, max_per_theme=3)
    assert result[0].paper_id == "0001"   # keep first
    assert result[1].paper_id == "0002"   # high-score second
    assert all(p.paper_id != "0003" for p in result)  # low excluded


def test_select_cross_theme_candidates_respects_max() -> None:
    papers = {str(i): _paper(str(i), overall_score=4, startup_potential=4) for i in range(10)}
    theme = _theme("T", list(papers.keys()))
    result = _select_cross_theme_candidates(theme, papers, max_per_theme=2)
    assert len(result) <= 2


def test_select_cross_theme_candidates_empty_when_all_low_score() -> None:
    p = _paper("0001", overall_score=2, startup_potential=2)
    theme = _theme("T", ["0001"])
    result = _select_cross_theme_candidates(theme, {"0001": p})
    assert result == []


def test_cross_theme_csv_columns_has_required_fields() -> None:
    required = [
        "id", "theme_name", "themes_involved", "paper_ids", "title",
        "composite_score", "simple_summary", "future_importance",
        "personal_excitement", "total_priority_score",
    ]
    for col in required:
        assert col in CROSS_THEME_CSV_COLUMNS


def test_cross_theme_intersections_nonempty() -> None:
    assert len(CROSS_THEME_INTERSECTIONS) >= 4
    for name, themes in CROSS_THEME_INTERSECTIONS:
        assert isinstance(name, str) and name
        assert len(themes) >= 2


def test_build_cross_theme_prompt_structure() -> None:
    p1 = _paper("0001", title="World Model Paper")
    p2 = _paper("0002", title="Web Agent Paper")
    candidates = {
        "World Models & Simulation": [p1],
        "Web & Computer-Use Agents": [p2],
    }
    themes_by_name = {
        "World Models & Simulation": _theme("World Models & Simulation", ["0001"], "World models summary."),
        "Web & Computer-Use Agents": _theme("Web & Computer-Use Agents", ["0002"], "Web agents summary."),
    }
    messages = build_cross_theme_prompt("World Models × Web Agents", candidates, themes_by_name)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    content = messages[1]["content"]
    assert "World Model Paper" in content
    assert "Web Agent Paper" in content
    assert "0001" in content
    assert "0002" in content


def test_build_cross_theme_prompt_marks_keeps() -> None:
    p = _paper("0001", claude_final_label="keep")
    candidates = {"Robotics & Embodied AI": [p]}
    themes_by_name = {"Robotics & Embodied AI": _theme("Robotics & Embodied AI", ["0001"])}
    messages = build_cross_theme_prompt("Test", candidates, themes_by_name)
    assert "KEEP" in messages[1]["content"]


def _valid_cross_theme_response(paper_ids: list[str] | None = None) -> str:
    return json.dumps({
        "title": "Cross-Theme Startup",
        "core_capability": "Does something novel across themes.",
        "added_value": "Better together.",
        "target_user": "ML engineer at mid-size company.",
        "problem_statement": "Current tools don't cross themes well.",
        "wedge_description": "Start with one vertical.",
        "risks": "Hard to copy but technically risky.",
        "paper_ids": paper_ids or ["0001", "0002"],
        "themes_involved": ["Theme A", "Theme B"],
        "future_importance": 5,
        "personal_excitement": 4,
    })


def test_parse_cross_theme_response_valid() -> None:
    raw = _valid_cross_theme_response(["0001", "0002"])
    result = parse_cross_theme_response(raw, {"0001", "0002", "0003"})
    assert result is not None
    assert result["title"] == "Cross-Theme Startup"
    assert result["paper_ids"] == ["0001", "0002"]


def test_parse_cross_theme_response_filters_invalid_ids() -> None:
    raw = _valid_cross_theme_response(["0001", "9999"])  # 9999 not in valid set
    result = parse_cross_theme_response(raw, {"0001", "0002"})
    # Only "0001" is valid — fewer than 2 → None
    assert result is None


def test_parse_cross_theme_response_skip() -> None:
    raw = json.dumps({"skip": True, "reason": "No compelling combination."})
    result = parse_cross_theme_response(raw, {"0001", "0002"})
    assert result is None


def test_parse_cross_theme_response_missing_fields() -> None:
    raw = json.dumps({"title": "Only a title"})
    result = parse_cross_theme_response(raw, {"0001", "0002"})
    assert result is None


def test_parse_cross_theme_response_invalid_json() -> None:
    result = parse_cross_theme_response("not json at all", {"0001"})
    assert result is None


# ── generate_cross_theme_composites (integration with mocked LLM) ──────────

def _make_cross_theme_llm_fn(paper_ids: list[str]) -> Callable[[list[dict]], str]:
    """Returns a mock LLM fn that gives a valid cross-theme idea, then scores, then summary."""
    from collections import deque

    # Response queue: cross-theme idea → scoring → simple summary
    responses: deque = deque([
        _valid_cross_theme_response(paper_ids),
        json.dumps({
            "wedge_clarity": 4, "technical_moat": 4, "market_pull": 4,
            "founder_fit": 4, "composite_synergy": 4, "scoring_notes": "Good.",
        }),
        json.dumps({"simple_summary": "This is for engineers. It does great things. Very exciting."}),
    ])

    def fn(messages: list[dict]) -> str:
        if responses:
            return responses.popleft()
        return json.dumps({"simple_summary": "Fallback summary."})

    return fn


def test_generate_cross_theme_composites_produces_idea() -> None:
    p1 = _paper("0001", overall_score=4, startup_potential=4, claude_final_label="keep")
    p2 = _paper("0002", overall_score=4, startup_potential=4)
    papers_by_id = {"0001": p1, "0002": p2}

    themes = [
        _theme("World Models & Simulation", ["0001"]),
        _theme("Web & Computer-Use Agents", ["0002"]),
    ]
    # Only first intersection matches; limit to 1
    llm_fn = _make_cross_theme_llm_fn(["0001", "0002"])
    results = generate_cross_theme_composites(papers_by_id, themes, llm_fn, max_ideas=1)

    assert len(results) == 1
    row = results[0]
    assert row["theme_name"] == "CROSS-THEME"
    assert row["title"] == "Cross-Theme Startup"
    assert row["future_importance"] == 5
    assert row["personal_excitement"] == 4
    assert isinstance(row["total_priority_score"], float)
    assert row["simple_summary"] != "[summary unavailable]"


def test_generate_cross_theme_composites_total_priority_formula() -> None:
    """total = 0.5*composite + 0.3*future + 0.2*excitement."""
    p1 = _paper("0001", overall_score=4, startup_potential=4, claude_final_label="keep")
    p2 = _paper("0002", overall_score=4, startup_potential=4)
    papers_by_id = {"0001": p1, "0002": p2}
    themes = [
        _theme("World Models & Simulation", ["0001"]),
        _theme("Web & Computer-Use Agents", ["0002"]),
    ]
    llm_fn = _make_cross_theme_llm_fn(["0001", "0002"])
    results = generate_cross_theme_composites(papers_by_id, themes, llm_fn, max_ideas=1)

    row = results[0]
    cs = float(row["composite_score"])
    fi = float(row["future_importance"])
    pe = float(row["personal_excitement"])
    expected = round(0.5 * cs + 0.3 * fi + 0.2 * pe, 2)
    assert abs(float(row["total_priority_score"]) - expected) < 0.01


def test_generate_cross_theme_composites_skips_when_too_few_themes() -> None:
    # Only one theme available — no cross-theme idea possible
    p1 = _paper("0001", overall_score=4, startup_potential=4)
    papers_by_id = {"0001": p1}
    themes = [_theme("World Models & Simulation", ["0001"])]

    called = []
    def llm_fn(messages: list[dict]) -> str:
        called.append(1)
        return _valid_cross_theme_response(["0001", "0002"])

    results = generate_cross_theme_composites(papers_by_id, themes, llm_fn, max_ideas=8)
    # No calls should be made since no intersection has 2+ themes with candidates
    assert len(results) == 0


def test_load_themes_from_csv_basic(tmp_path: Path) -> None:
    themes_path = tmp_path / "themes.csv"
    rows = [
        {"theme_name": "Robotics", "keywords": "robot, manipulation",
         "paper_ids": '["0001","0002"]', "summary": "Robots doing things.", "num_papers": "2"},
    ]
    with themes_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    loaded = load_themes_from_csv(str(themes_path))
    assert len(loaded) == 1
    assert loaded[0].name == "Robotics"
    assert loaded[0].paper_ids == ["0001", "0002"]
    assert loaded[0].summary == "Robots doing things."


def test_load_themes_from_csv_missing_file(tmp_path: Path) -> None:
    result = load_themes_from_csv(str(tmp_path / "nonexistent.csv"))
    assert result == []


# Need Callable for the test helper
from typing import Callable
