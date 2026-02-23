"""CSV file sink for the Deep-Tech Idea Pipeline."""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from models import Paper

CSV_OUTPUT_PATH = os.getenv("CSV_OUTPUT_PATH", "papers_pipeline.csv")

LOGGER = logging.getLogger(__name__)

CSV_COLUMNS = [
    "paper_id",
    "title",
    "url",
    "abstract",
    "published_at",
    "summary_problem",
    "summary_core_method",
    "summary_key_technical_idea",
    "summary_inputs_outputs",
    "summary_data_assumptions",
    "summary_metrics_and_baselines",
    "summary_limitations",
    "capability_plain_language_capability",
    "product_angles",
    "competition",
    "top_bets",
    "created_at",
]


def paper_already_exists(paper: Paper) -> bool:
    """Return True if a row with paper.paper_id already exists in the CSV."""
    path = Path(CSV_OUTPUT_PATH)
    if not path.exists():
        return False

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("paper_id") == paper.paper_id:
                return True
    return False


def write_paper_entry(paper: Paper, analysis: dict[str, Any]) -> None:
    """Append a row to the CSV (creating it with a header if needed)."""
    path = Path(CSV_OUTPUT_PATH)
    write_header = not path.exists() or path.stat().st_size == 0

    summary = analysis.get("summary", {}) if isinstance(analysis.get("summary"), dict) else {}
    capability = analysis.get("capability", {}) if isinstance(analysis.get("capability"), dict) else {}

    row = {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "url": paper.url,
        "abstract": paper.abstract or "",
        "published_at": paper.published_at.isoformat(),
        "summary_problem": _as_text(summary.get("problem")),
        "summary_core_method": _as_text(summary.get("core_method")),
        "summary_key_technical_idea": _as_text(summary.get("key_technical_idea")),
        "summary_inputs_outputs": _as_text(summary.get("inputs_outputs")),
        "summary_data_assumptions": _as_text(summary.get("data_assumptions")),
        "summary_metrics_and_baselines": _as_text(summary.get("metrics_and_baselines")),
        "summary_limitations": _as_text(summary.get("limitations")),
        "capability_plain_language_capability": _as_text(capability.get("plain_language_capability")),
        "product_angles": json.dumps(analysis.get("product_angles", [])),
        "competition": json.dumps(analysis.get("competition", [])),
        "top_bets": json.dumps(analysis.get("top_bets", [])),
        "created_at": datetime.now(UTC).isoformat(),
    }

    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    LOGGER.info("Wrote CSV row for paper_id=%s to %s", paper.paper_id, CSV_OUTPUT_PATH)


def _as_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
