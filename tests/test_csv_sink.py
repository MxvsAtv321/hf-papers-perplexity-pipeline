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
