"""CLI entrypoint for the daily Deep-Tech Idea Pipeline."""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from hf_feed import fetch_papers
from notion_client import create_paper_entry, paper_already_exists
from perplexity_client import analyze_paper


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


def run(limit: int | None, dry_run: bool) -> None:
    """Run one full pipeline cycle."""
    papers = fetch_papers(limit=limit)
    logging.info("Fetched %s papers from Hugging Face Daily Papers", len(papers))

    processed = 0
    skipped = 0
    failed = 0

    for paper in papers:
        logging.info("Evaluating paper: %s", paper.title)
        if paper_already_exists(paper.paper_id):
            skipped += 1
            logging.info("Skipping existing paper_id=%s", paper.paper_id)
            continue

        if dry_run:
            processed += 1
            logging.info("[dry-run] Would analyze and create Notion entry: %s", paper.title)
            continue

        try:
            analysis = analyze_paper(paper)
            create_paper_entry(paper, analysis)
            processed += 1
            logging.info("Created Notion entry for paper_id=%s", paper.paper_id)
        except Exception as exc:  # broad by design to keep daily run resilient
            failed += 1
            logging.exception("Failed processing paper_id=%s: %s", paper.paper_id, exc)

    logging.info("Run complete. processed=%s skipped=%s failed=%s", processed, skipped, failed)


def main() -> None:
    """Initialize config and execute the pipeline."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
