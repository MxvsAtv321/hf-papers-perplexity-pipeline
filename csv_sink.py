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
    # Stage tracking — which pipeline stages were executed for this row
    "stage1_passed",    # survived date window + heuristic pre-filter
    "stage2_scored",    # OpenAI startup scoring completed
    "stage3_debated",   # Claude two-agent debate invoked (gate passed)
    "stage4_analyzed",  # deep OpenAI triage completed
    # Stage 2 — OpenAI startup scores
    "startup_potential",
    "market_pull",
    "technical_moat",
    "story_for_accelerator",
    "overall_score",
    "score_rationale",
    # Stage 3 — Claude two-agent debate (human-readable columns)
    "claude_tf_score",              # Technical Founder agent score (1–5)
    "claude_tf_rationale",          # Technical Founder rationale (≤400 chars)
    "claude_ap_score",              # Accelerator Partner agent score (1–5)
    "claude_ap_rationale",          # Accelerator Partner rationale (≤400 chars)
    "claude_final_score",           # Converged final score (1–5)
    "claude_final_label",           # keep | maybe | drop | unknown
    "claude_final_reason",          # Final verdict reason (≤400 chars)
    "claude_score_gap",             # |TF score − AP score|; empty if either is None
    "claude_disagreement_flag",     # True if agents disagreed strongly (gap≥2 or 4-vs-≤2)
    "claude_verdict_raw",           # Full JSON verdict for programmatic access
    # Stage 4 — Full OpenAI analysis
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


def write_paper_entry(
    paper: Paper,
    analysis: dict[str, Any],
    score: dict[str, Any] | None = None,
    debate: dict[str, Any] | None = None,
    stages: dict[str, bool] | None = None,
) -> None:
    """Append a row to the CSV (creating it with a header if needed).

    Args:
        paper:    Normalized paper record.
        analysis: Full OpenAI analysis dict (summary, product_angles, etc.).
        score:    Optional Stage-2 OpenAI scoring dict.
        debate:   Optional Stage-3 Claude debate verdict dict (normalized).
        stages:   Optional dict of stage flags (stage1_passed … stage4_analyzed).
    """
    path = Path(CSV_OUTPUT_PATH)
    write_header = not path.exists() or path.stat().st_size == 0

    summary = analysis.get("summary", {}) if isinstance(analysis.get("summary"), dict) else {}
    capability = analysis.get("capability", {}) if isinstance(analysis.get("capability"), dict) else {}

    score = score or {}
    debate = debate or {}
    stages = stages or {}

    tf = debate.get("technical_founder") or {}
    ap = debate.get("accelerator_partner") or {}
    fv = debate.get("final_verdict") or {}

    # Disagreement signal — only computable when both agents scored
    tf_score = tf.get("score")
    ap_score = ap.get("score")
    if isinstance(tf_score, int) and isinstance(ap_score, int):
        gap: int | str = abs(tf_score - ap_score)
        disagreement: bool | str = gap >= 2 or (
            max(tf_score, ap_score) >= 4 and min(tf_score, ap_score) <= 2
        )
    else:
        gap = ""
        disagreement = ""

    row = {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "url": paper.url,
        "abstract": paper.abstract or "",
        "published_at": paper.published_at.isoformat(),
        # Stage flags
        "stage1_passed": stages.get("stage1_passed", False),
        "stage2_scored": stages.get("stage2_scored", False),
        "stage3_debated": stages.get("stage3_debated", False),
        "stage4_analyzed": stages.get("stage4_analyzed", False),
        # Stage 2 scores
        "startup_potential": score.get("startup_potential", ""),
        "market_pull": score.get("market_pull", ""),
        "technical_moat": score.get("technical_moat", ""),
        "story_for_accelerator": score.get("story_for_accelerator", ""),
        "overall_score": score.get("overall_score", ""),
        "score_rationale": _as_text(score.get("rationale"), max_len=400),
        # Stage 3 Claude debate — human-readable
        "claude_tf_score": "" if tf_score is None else tf_score,
        "claude_tf_rationale": _as_text(tf.get("rationale"), max_len=400),
        "claude_ap_score": "" if ap_score is None else ap_score,
        "claude_ap_rationale": _as_text(ap.get("rationale"), max_len=400),
        "claude_final_score": "" if fv.get("score") is None else fv.get("score"),
        "claude_final_label": _as_text(fv.get("label")),
        "claude_final_reason": _as_text(fv.get("reason"), max_len=400),
        "claude_score_gap": gap,
        "claude_disagreement_flag": disagreement,
        "claude_verdict_raw": json.dumps(debate) if debate else "",
        # Stage 4 full analysis
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


def _as_text(value: Any, max_len: int = 500) -> str:
    """Convert value to a stripped string, truncated to max_len chars."""
    s = value.strip() if isinstance(value, str) else ""
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s
