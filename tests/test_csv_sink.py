from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import csv_sink
from models import Paper

SAMPLE_PAPER = Paper(
    paper_id="2501.99999",
    title="Test Paper",
    url="https://huggingface.co/papers/2501.99999",
    abstract="A test abstract.",
    published_at=datetime(2026, 2, 22, 12, 0, 0, tzinfo=UTC),
)

SAMPLE_ANALYSIS = {
    "summary": {
        "problem": "The problem",
        "core_method": "The method",
        "key_technical_idea": "Key idea",
        "inputs_outputs": "Inputs and outputs",
        "data_assumptions": "Assumptions",
        "metrics_and_baselines": "BLEU score",
        "limitations": "None noted",
    },
    "capability": {"plain_language_capability": "It can do X"},
    "product_angles": [{"name": "Angle A", "target_user_persona": "Devs"}],
    "competition": [{"name": "Competitor X", "type": "SaaS", "url": "https://example.com", "difference_vs_paper": "Less accurate"}],
    "top_bets": [{"product_angle_name": "Angle A", "rationale": "Large market"}],
}

SAMPLE_STAGES = {
    "stage1_passed": True,
    "stage2_scored": True,
    "stage3_debated": False,
    "stage4_analyzed": True,
}


@pytest.fixture(autouse=True)
def patch_csv_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point CSV_OUTPUT_PATH at a temp file for every test."""
    output = tmp_path / "test_output.csv"
    monkeypatch.setattr(csv_sink, "CSV_OUTPUT_PATH", str(output))


def test_paper_already_exists_false_when_no_file() -> None:
    assert csv_sink.paper_already_exists(SAMPLE_PAPER) is False


def test_write_paper_entry_creates_file_with_header(tmp_path: Path) -> None:
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS)

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    assert path.exists()

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["paper_id"] == "2501.99999"
    assert row["title"] == "Test Paper"
    assert row["summary_problem"] == "The problem"
    assert row["capability_plain_language_capability"] == "It can do X"


def test_write_paper_entry_serializes_lists() -> None:
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS)

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        row = next(reader)

    product_angles = json.loads(row["product_angles"])
    assert isinstance(product_angles, list)
    assert product_angles[0]["name"] == "Angle A"

    top_bets = json.loads(row["top_bets"])
    assert top_bets[0]["rationale"] == "Large market"


def test_paper_already_exists_true_after_write() -> None:
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS)
    assert csv_sink.paper_already_exists(SAMPLE_PAPER) is True


def test_paper_already_exists_false_for_different_id() -> None:
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS)

    other = Paper(
        paper_id="9999.00001",
        title="Other Paper",
        url="https://huggingface.co/papers/9999.00001",
        abstract="",
        published_at=datetime(2026, 2, 22, tzinfo=UTC),
    )
    assert csv_sink.paper_already_exists(other) is False


def test_write_paper_entry_with_score_populates_score_columns() -> None:
    score = {
        "startup_potential": 4,
        "market_pull": 3,
        "technical_moat": 5,
        "story_for_accelerator": 4,
        "overall_score": 4,
        "rationale": "Strong technical moat.",
    }
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS, score=score)

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))

    assert row["startup_potential"] == "4"
    assert row["overall_score"] == "4"
    assert row["score_rationale"] == "Strong technical moat."


def test_write_paper_entry_without_score_leaves_score_columns_empty() -> None:
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS)

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))

    assert row["overall_score"] == ""
    assert row["score_rationale"] == ""


def test_write_paper_entry_with_debate_populates_claude_columns() -> None:
    debate = {
        "technical_founder": {"score": 4, "rationale": "Strong moat."},
        "accelerator_partner": {"score": 5, "rationale": "Clear market pull."},
        "final_verdict": {"score": 4, "label": "keep", "reason": "High confidence."},
    }
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS, debate=debate)

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))

    assert row["claude_tf_score"] == "4"
    assert row["claude_tf_rationale"] == "Strong moat."
    assert row["claude_ap_score"] == "5"
    assert row["claude_ap_rationale"] == "Clear market pull."
    assert row["claude_final_label"] == "keep"
    assert row["claude_final_score"] == "4"
    assert row["claude_final_reason"] == "High confidence."


def test_write_paper_entry_without_debate_leaves_claude_columns_empty() -> None:
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS)

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))

    assert row["claude_tf_score"] == ""
    assert row["claude_ap_score"] == ""
    assert row["claude_final_label"] == ""


def test_write_paper_entry_appends_multiple_rows() -> None:
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS)

    second = Paper(
        paper_id="2501.88888",
        title="Second Paper",
        url="https://huggingface.co/papers/2501.88888",
        abstract="Another abstract.",
        published_at=datetime(2026, 2, 22, tzinfo=UTC),
    )
    csv_sink.write_paper_entry(second, SAMPLE_ANALYSIS)

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 2
    assert rows[0]["paper_id"] == "2501.99999"
    assert rows[1]["paper_id"] == "2501.88888"


# ---------------------------------------------------------------------------
# Stage flag tests
# ---------------------------------------------------------------------------

def test_stage_flags_written_correctly() -> None:
    stages = {
        "stage1_passed": True,
        "stage2_scored": True,
        "stage3_debated": False,
        "stage4_analyzed": True,
    }
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS, stages=stages)

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))

    assert row["stage1_passed"] == "True"
    assert row["stage2_scored"] == "True"
    assert row["stage3_debated"] == "False"
    assert row["stage4_analyzed"] == "True"


def test_stage_flags_default_to_false_when_not_passed() -> None:
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS)  # no stages

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))

    assert row["stage1_passed"] == "False"
    assert row["stage3_debated"] == "False"


def test_stage3_false_when_debate_gate_not_passed() -> None:
    """Paper that doesn't clear the debate gate should have stage3_debated=False."""
    stages = {
        "stage1_passed": True,
        "stage2_scored": True,
        "stage3_debated": False,   # gate rejected — no debate
        "stage4_analyzed": True,
    }
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS, stages=stages)

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))

    assert row["stage3_debated"] == "False"
    assert row["claude_tf_score"] == ""      # no debate, so no Claude scores
    assert row["claude_final_label"] == ""


def test_stage3_true_when_debate_invoked() -> None:
    """Paper that passes the gate has stage3_debated=True and Claude columns populated."""
    debate = {
        "technical_founder": {"score": 4, "rationale": "Solid moat."},
        "accelerator_partner": {"score": 4, "rationale": "Good market."},
        "final_verdict": {"score": 4, "label": "keep", "reason": "Strong pick."},
    }
    stages = {
        "stage1_passed": True,
        "stage2_scored": True,
        "stage3_debated": True,
        "stage4_analyzed": True,
    }
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS, debate=debate, stages=stages)

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))

    assert row["stage3_debated"] == "True"
    assert row["claude_tf_score"] == "4"
    assert row["claude_final_label"] == "keep"


# ---------------------------------------------------------------------------
# claude_score_gap and claude_disagreement_flag tests
# ---------------------------------------------------------------------------

def test_claude_score_gap_computed_when_both_agents_scored() -> None:
    debate = {
        "technical_founder": {"score": 5, "rationale": "Very strong."},
        "accelerator_partner": {"score": 2, "rationale": "Weak market."},
        "final_verdict": {"score": 3, "label": "maybe", "reason": "Mixed signals."},
    }
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS, debate=debate)

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))

    assert row["claude_score_gap"] == "3"
    assert row["claude_disagreement_flag"] == "True"


def test_claude_score_gap_zero_when_agents_agree() -> None:
    debate = {
        "technical_founder": {"score": 4, "rationale": "Good."},
        "accelerator_partner": {"score": 4, "rationale": "Agree."},
        "final_verdict": {"score": 4, "label": "keep", "reason": "Consensus."},
    }
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS, debate=debate)

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))

    assert row["claude_score_gap"] == "0"
    assert row["claude_disagreement_flag"] == "False"


def test_claude_score_gap_empty_when_scores_are_none() -> None:
    debate = {
        "technical_founder": {"score": None, "rationale": "claude_not_configured"},
        "accelerator_partner": {"score": None, "rationale": "claude_not_configured"},
        "final_verdict": {"score": None, "label": "unknown", "reason": "no key"},
    }
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS, debate=debate)

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))

    assert row["claude_score_gap"] == ""
    assert row["claude_disagreement_flag"] == ""


def test_claude_verdict_raw_is_json() -> None:
    debate = {
        "technical_founder": {"score": 4, "rationale": "ok"},
        "accelerator_partner": {"score": 4, "rationale": "ok"},
        "final_verdict": {"score": 4, "label": "keep", "reason": "ok"},
    }
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS, debate=debate)

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))

    raw = json.loads(row["claude_verdict_raw"])
    assert raw["final_verdict"]["label"] == "keep"


def test_long_rationale_is_truncated_in_csv() -> None:
    long_text = "X" * 600
    debate = {
        "technical_founder": {"score": 4, "rationale": long_text},
        "accelerator_partner": {"score": 4, "rationale": "Short."},
        "final_verdict": {"score": 4, "label": "keep", "reason": "ok"},
    }
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS, debate=debate)

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))

    assert len(row["claude_tf_rationale"]) <= 400


def test_disagreement_flag_true_when_one_very_high_other_very_low() -> None:
    """Gap < 2 but 4-vs-2 should still flag disagreement (4 >= 4 and 2 <= 2)."""
    debate = {
        "technical_founder": {"score": 4, "rationale": "Likes it."},
        "accelerator_partner": {"score": 2, "rationale": "Skeptical."},
        "final_verdict": {"score": 3, "label": "maybe", "reason": "Split verdict."},
    }
    csv_sink.write_paper_entry(SAMPLE_PAPER, SAMPLE_ANALYSIS, debate=debate)

    path = Path(csv_sink.CSV_OUTPUT_PATH)
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))

    # gap = 2 → flag from gap condition
    assert row["claude_disagreement_flag"] == "True"
