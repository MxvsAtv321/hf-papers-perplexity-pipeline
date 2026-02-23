"""Post-run reporting: generates focused summary tables from papers_pipeline.csv.

Two output files are produced on every non-dry-run:

  papers_debated.csv   — papers that cleared the Claude debate gate, sorted by
                         final score descending.  Clean columns only — no JSON
                         blobs, no long-text noise — so it opens readably in
                         Excel / Numbers / Google Sheets.

  papers_top_picks.csv — top TOP_PICKS_N papers by composite score across ALL
                         pipeline rows (debated or not), sorted best-first.
                         One row = one startup opportunity.

Both files are also runnable standalone:
    python report.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable paths / limits
# ---------------------------------------------------------------------------

_DEFAULT_SOURCE_CSV = os.getenv("CSV_OUTPUT_PATH", "papers_pipeline.csv")
DEBATED_REPORT_PATH = os.getenv("DEBATED_REPORT_PATH", "papers_debated.csv")
TOP_PICKS_REPORT_PATH = os.getenv("TOP_PICKS_REPORT_PATH", "papers_top_picks.csv")
TOP_PICKS_N = int(os.getenv("TOP_PICKS_N", "15"))

# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

DEBATED_COLUMNS = [
    "rank",
    "title",
    "url",
    # OpenAI startup scores — the cheap pre-filter signal
    "openai_overall",
    "openai_startup",
    "openai_market",
    "openai_moat",
    "openai_story",
    # Agent 1 — Technical Founder perspective
    "tf_score",
    "tf_rationale",
    # Agent 2 — Accelerator Partner perspective
    "ap_score",
    "ap_rationale",
    # Converged verdict
    "final_score",
    "final_label",
    "final_reason",
    # Did they agree?
    "score_gap",
    "disagreement",
    # What the paper does
    "problem_solved",
    "capability",
    "best_product_angle",
]

TOP_PICKS_COLUMNS = [
    "rank",
    "title",
    "url",
    "composite_score",
    "final_label",
    "problem_solved",
    "capability",
    "best_product_angle",
    "target_user",
    "tf_take",
    "ap_take",
    "final_verdict",
    "openai_rationale",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _first_product_angle(product_angles_json: str) -> tuple[str, str]:
    """Return (name, target_user_persona) of the first product angle, or ('', '')."""
    try:
        angles = json.loads(product_angles_json or "[]")
        if angles and isinstance(angles, list) and isinstance(angles[0], dict):
            return (
                angles[0].get("name", ""),
                angles[0].get("target_user_persona", ""),
            )
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    return ("", "")


def _composite_score(row: dict) -> float:
    """Weighted blend: 65% Claude final score + 35% OpenAI overall.

    Falls back to whichever signal is available if the other is missing.
    """
    openai = _safe_float(row.get("overall_score"))
    claude = _safe_float(row.get("claude_final_score"))

    if openai is None and claude is None:
        return 0.0
    if claude is None:
        return openai  # type: ignore[return-value]
    if openai is None:
        return claude
    return round(0.65 * claude + 0.35 * openai, 2)


def _write_csv(path: str, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------


def _build_debated_rows(rows: list[dict]) -> list[dict]:
    debated = [r for r in rows if str(r.get("stage3_debated", "")).lower() == "true"]
    debated.sort(
        key=lambda r: (_safe_float(r.get("claude_final_score")) or 0),
        reverse=True,
    )
    out = []
    for i, r in enumerate(debated, 1):
        angle_name, _ = _first_product_angle(r.get("product_angles", ""))
        out.append({
            "rank": i,
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "openai_overall": r.get("overall_score", ""),
            "openai_startup": r.get("startup_potential", ""),
            "openai_market": r.get("market_pull", ""),
            "openai_moat": r.get("technical_moat", ""),
            "openai_story": r.get("story_for_accelerator", ""),
            "tf_score": r.get("claude_tf_score", ""),
            "tf_rationale": r.get("claude_tf_rationale", ""),
            "ap_score": r.get("claude_ap_score", ""),
            "ap_rationale": r.get("claude_ap_rationale", ""),
            "final_score": r.get("claude_final_score", ""),
            "final_label": r.get("claude_final_label", ""),
            "final_reason": r.get("claude_final_reason", ""),
            "score_gap": r.get("claude_score_gap", ""),
            "disagreement": r.get("claude_disagreement_flag", ""),
            "problem_solved": r.get("summary_problem", ""),
            "capability": r.get("capability_plain_language_capability", ""),
            "best_product_angle": angle_name,
        })
    return out


def _build_top_picks_rows(rows: list[dict]) -> list[dict]:
    scored = sorted(rows, key=_composite_score, reverse=True)
    top = scored[:TOP_PICKS_N]
    out = []
    for i, r in enumerate(top, 1):
        cs = _composite_score(r)
        angle_name, target_user = _first_product_angle(r.get("product_angles", ""))
        label = r.get("claude_final_label", "")
        # For papers without debate, fall back to OpenAI rationale as verdict text
        verdict_text = r.get("claude_final_reason", "") or r.get("score_rationale", "")
        out.append({
            "rank": i,
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "composite_score": cs,
            "final_label": label or "—",
            "problem_solved": r.get("summary_problem", ""),
            "capability": r.get("capability_plain_language_capability", ""),
            "best_product_angle": angle_name,
            "target_user": target_user,
            "tf_take": r.get("claude_tf_rationale", ""),
            "ap_take": r.get("claude_ap_rationale", ""),
            "final_verdict": verdict_text,
            "openai_rationale": r.get("score_rationale", ""),
        })
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_reports(source_csv: str | None = None) -> None:
    """Read the main pipeline CSV and write debated + top-picks report files."""
    source = source_csv or _DEFAULT_SOURCE_CSV
    path = Path(source)

    if not path.exists() or path.stat().st_size == 0:
        LOGGER.warning("report: source CSV not found or empty: %s", source)
        return

    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        LOGGER.warning("report: source CSV has no data rows")
        return

    debated_rows = _build_debated_rows(rows)
    _write_csv(DEBATED_REPORT_PATH, DEBATED_COLUMNS, debated_rows)
    LOGGER.info(
        "report: %d debated papers → %s", len(debated_rows), DEBATED_REPORT_PATH
    )

    top_rows = _build_top_picks_rows(rows)
    _write_csv(TOP_PICKS_REPORT_PATH, TOP_PICKS_COLUMNS, top_rows)
    LOGGER.info(
        "report: top %d papers → %s", len(top_rows), TOP_PICKS_REPORT_PATH
    )


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    source = sys.argv[1] if len(sys.argv) > 1 else None
    generate_reports(source)
    print(f"Debated papers  → {DEBATED_REPORT_PATH}")
    print(f"Top {TOP_PICKS_N} picks    → {TOP_PICKS_REPORT_PATH}")
