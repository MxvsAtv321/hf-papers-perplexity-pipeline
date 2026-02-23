"""CLI entrypoint for the daily Deep-Tech Idea Pipeline."""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

from claude_agentic import debate_paper_with_two_agents
from csv_sink import paper_already_exists, write_paper_entry
from filters import is_potential_startup_paper
from hf_feed import fetch_papers
from llm_client import analyze_paper, score_paper_for_startup


def parse_args() -> argparse.Namespace:
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(description="Run the Hugging Face -> Perplexity -> Notion pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of recent papers to process")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be processed, without API writes",
    )
    return parser.parse_args()


def _should_run_debate(score: dict) -> bool:
    """Return True only when the paper clears both overall and commercial thresholds.

    Two conditions must both hold:
    1. overall_score >= AGENTIC_MIN_OVERALL_SCORE (default 4) — filters weak papers.
    2. startup_potential >= 4 OR market_pull >= 4 — at least one strong commercial
       signal. Pure technical novelty (high technical_moat alone) is not sufficient;
       there must be evidence the paper can become a company.
    """
    min_overall = int(os.getenv("AGENTIC_MIN_OVERALL_SCORE", "4"))

    overall = score.get("overall_score")
    if not isinstance(overall, int) or overall < min_overall:
        return False

    startup = score.get("startup_potential")
    market = score.get("market_pull")
    return (isinstance(startup, int) and startup >= 4) or (
        isinstance(market, int) and market >= 4
    )


def run(limit: int | None, dry_run: bool) -> None:
    """Run one full pipeline cycle."""
    papers = fetch_papers(limit=limit)
    logging.info("Fetched %s papers from Hugging Face Daily Papers", len(papers))

    # Stage 1: heuristic pre-filter — no LLM calls, no cost.
    viable = [p for p in papers if is_potential_startup_paper(p)]
    logging.info(
        "Heuristic filter: total=%s, viable=%s, dropped=%s",
        len(papers),
        len(viable),
        len(papers) - len(viable),
    )

    processed = 0
    skipped = 0
    failed = 0

    for paper in viable:
        logging.info("Evaluating paper: %s", paper.title)
        if paper_already_exists(paper):
            skipped += 1
            logging.info("Skipping existing paper_id=%s", paper.paper_id)
            continue

        if dry_run:
            processed += 1
            logging.info("[dry-run] Would process: %s", paper.title)
            continue

        # Per-paper stage tracking — all flags start False; set to True on success.
        # stage1_passed is always True at this point (heuristic filter already ran).
        stages: dict[str, bool] = {
            "stage1_passed": True,
            "stage2_scored": False,
            "stage3_debated": False,
            "stage4_analyzed": False,
        }

        try:
            # Stage 2: cheap OpenAI scoring
            score = score_paper_for_startup(paper)
            stages["stage2_scored"] = True

            # Stage 3: Claude two-agent debate — only for commercially-viable high scorers.
            # Requires overall_score >= AGENTIC_MIN_OVERALL_SCORE AND at least one of
            # startup_potential or market_pull >= 4. This avoids spending two Claude API
            # calls on papers that are technically impressive but lack a commercial angle.
            debate: dict | None = None
            if _should_run_debate(score):
                debate = debate_paper_with_two_agents(paper, score)
                stages["stage3_debated"] = True
                logging.info("Claude debate completed for paper_id=%s", paper.paper_id)
            else:
                logging.info(
                    "Claude debate skipped for paper_id=%s: "
                    "overall_score=%s startup_potential=%s market_pull=%s",
                    paper.paper_id,
                    score.get("overall_score"),
                    score.get("startup_potential"),
                    score.get("market_pull"),
                )

            # Stage 4: Full OpenAI analysis
            analysis = analyze_paper(paper)
            stages["stage4_analyzed"] = True

            write_paper_entry(paper, analysis, score=score, debate=debate, stages=stages)
            processed += 1
            logging.info("Wrote CSV entry for paper_id=%s", paper.paper_id)
        except Exception as exc:  # broad by design to keep daily run resilient
            failed += 1
            logging.exception("Failed processing paper_id=%s: %s", paper.paper_id, exc)

    logging.info("Run complete. processed=%s skipped=%s failed=%s", processed, skipped, failed)

    # Post-run: regenerate focused report tables from the full CSV
    if not dry_run:
        try:
            from report import generate_reports  # noqa: PLC0415
            generate_reports()
        except Exception as exc:
            logging.warning("Report generation failed (non-fatal): %s", exc)


def main() -> None:
    """Initialize config and execute the pipeline."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
