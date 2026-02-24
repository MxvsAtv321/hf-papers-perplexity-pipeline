"""CLI entrypoint for the daily Deep-Tech Idea Pipeline."""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

from claude_agentic import debate_paper_with_two_agents
from csv_sink import WIDE_SCOUT_CSV_PATH, paper_already_exists, write_paper_entry
from filters import is_potential_startup_paper
from hf_feed import fetch_papers
from llm_client import analyze_paper, score_paper_for_startup


def parse_args() -> argparse.Namespace:
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(description="Run the Hugging Face -> Perplexity -> Notion pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of recent papers to process (full mode only)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be processed, without API writes",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "wide_scout"],
        default="full",
        help=(
            "Pipeline mode. 'full' (default): fetch recent papers and run all stages. "
            "'wide_scout': broad-coverage sweep with capped LLM usage — fetches many days, "
            "runs Stage 2 on up to MAX_LLM_PAPERS_WIDE papers, Stage 3-4 on top MAX_DEEP_PAPERS_WIDE."
        ),
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


def run_wide_scout(
    max_fetch_days: int,
    max_llm_papers: int,
    max_deep_papers: int,
    dry_run: bool,
) -> None:
    """Broad-coverage, cost-capped pipeline sweep.

    Fetches up to max_fetch_days of HF papers, applies the heuristic filter and
    CSV dedup, then scores up to max_llm_papers papers with OpenAI (Stage 2).
    If max_deep_papers > 0, the top-scoring papers also get the Claude debate
    (Stage 3) and full OpenAI analysis (Stage 4). Results go to papers_wide_scout.csv
    (overridable via WIDE_SCOUT_CSV_PATH) so the main papers_pipeline.csv is untouched.

    Args:
        max_fetch_days: How many past days to query from the HF API.
        max_llm_papers: Maximum number of papers to run through Stage 2.
        max_deep_papers: Number of top-scoring papers that also get Stage 3-4.
            Set to 0 (default) to skip Stage 3-4 entirely in this mode.
        dry_run: When True, log what would be processed without any API calls or writes.
    """
    papers = fetch_papers(fetch_days=max_fetch_days)
    logging.info("Wide scout: fetched %s papers (%s-day window)", len(papers), max_fetch_days)

    # Stage 1: heuristic pre-filter
    viable = [p for p in papers if is_potential_startup_paper(p)]
    logging.info(
        "Wide scout heuristic: total=%s viable=%s dropped=%s",
        len(papers),
        len(viable),
        len(papers) - len(viable),
    )

    # Dedup against the wide scout CSV (separate from the full-mode CSV)
    new_papers = [p for p in viable if not paper_already_exists(p, csv_path=WIDE_SCOUT_CSV_PATH)]
    logging.info(
        "Wide scout dedup: %s new papers (skipped %s already in CSV)",
        len(new_papers),
        len(viable) - len(new_papers),
    )

    # Cap Stage 2 at max_llm_papers
    to_score = new_papers[:max_llm_papers]
    logging.info(
        "Wide scout: will score %s papers (cap=%s, total_new=%s)",
        len(to_score),
        max_llm_papers,
        len(new_papers),
    )

    if dry_run:
        for p in to_score:
            logging.info("[dry-run] Would score: %s", p.title)
        logging.info("[dry-run] Wide scout complete: would process %s papers", len(to_score))
        return

    # Stage 2: score all selected papers
    scored: list[tuple] = []
    stage2_failed = 0
    for paper in to_score:
        try:
            score = score_paper_for_startup(paper)
            scored.append((paper, score))
        except Exception as exc:
            stage2_failed += 1
            logging.exception("Wide scout: score failed for paper_id=%s: %s", paper.paper_id, exc)

    logging.info("Wide scout Stage 2: scored=%s failed=%s", len(scored), stage2_failed)

    # Determine which papers get Stage 3-4 (top N by overall_score)
    if max_deep_papers > 0:
        scored_sorted = sorted(
            scored,
            key=lambda ps: ps[1].get("overall_score") or 0,
            reverse=True,
        )
        deep_ids = {p.paper_id for p, _ in scored_sorted[:max_deep_papers]}
        logging.info(
            "Wide scout: running Stage 3-4 on top %s papers (max_deep_papers=%s)",
            len(deep_ids),
            max_deep_papers,
        )
    else:
        deep_ids = set()

    # Write all scored papers; deep papers also get Stage 3-4
    written = 0
    for paper, score in scored:
        stages: dict[str, bool] = {
            "stage1_passed": True,
            "stage2_scored": True,
            "stage3_debated": False,
            "stage4_analyzed": False,
        }
        debate: dict | None = None
        analysis: dict = {}

        if paper.paper_id in deep_ids:
            if _should_run_debate(score):
                try:
                    debate = debate_paper_with_two_agents(paper, score)
                    stages["stage3_debated"] = True
                except Exception as exc:
                    logging.exception(
                        "Wide scout: debate failed for paper_id=%s: %s", paper.paper_id, exc
                    )

            try:
                analysis = analyze_paper(paper)
                stages["stage4_analyzed"] = True
            except Exception as exc:
                logging.exception(
                    "Wide scout: analysis failed for paper_id=%s: %s", paper.paper_id, exc
                )

        try:
            write_paper_entry(
                paper,
                analysis,
                score=score,
                debate=debate,
                stages=stages,
                csv_path=WIDE_SCOUT_CSV_PATH,
            )
            written += 1
        except Exception as exc:
            logging.exception(
                "Wide scout: write failed for paper_id=%s: %s", paper.paper_id, exc
            )

    logging.info(
        "Wide scout complete: scored=%s written=%s stage2_failed=%s",
        len(scored),
        written,
        stage2_failed,
    )


def main() -> None:
    """Initialize config and execute the pipeline."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if args.mode == "wide_scout":
        max_fetch_days = int(os.getenv("MAX_FETCH_DAYS_WIDE", "180"))
        max_llm_papers = int(os.getenv("MAX_LLM_PAPERS_WIDE", "500"))
        max_deep_papers = int(os.getenv("MAX_DEEP_PAPERS_WIDE", "0"))
        run_wide_scout(
            max_fetch_days=max_fetch_days,
            max_llm_papers=max_llm_papers,
            max_deep_papers=max_deep_papers,
            dry_run=args.dry_run,
        )
    else:
        run(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
