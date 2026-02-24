"""CLI: mine emerging themes and composite startup ideas from pipeline CSVs.

Usage examples
--------------
# Inspect themes only (no LLM calls):
python mine_patterns.py

# With theme summaries from OpenAI:
python mine_patterns.py --generate-summaries

# Full run: themes + composite startup ideas:
python mine_patterns.py --generate-summaries --generate-composites

# Score previously generated composite ideas (reads papers_composite_ideas.csv):
python mine_patterns.py --score-composites

# Custom inputs:
python mine_patterns.py --input papers_debated.csv papers_wide_scout.csv --min-score 4

# Only papers with claude_final_label=keep:
python mine_patterns.py --keeps-only --generate-composites
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from pattern_miner import (
    COMPOSITES_CSV_COLUMNS,
    SCORED_COMPOSITES_CSV_COLUMNS,
    THEMES_CSV_COLUMNS,
    AnalyzedPaper,
    CompositeIdea,
    ScoredCompositeIdea,
    Theme,
    aggregate_paper_signals,
    build_composite_prompt,
    build_scoring_prompt,
    build_theme_summary_prompt,
    composite_ideas_to_rows,
    extract_themes,
    find_composite_candidates,
    load_composite_ideas,
    load_papers,
    parse_composite_response,
    parse_score_response,
    score_composite_ideas,
    scored_ideas_to_rows,
    themes_to_rows,
    write_csv,
)

LOGGER = logging.getLogger(__name__)

_DEFAULT_INPUTS = ["papers_wide_scout.csv", "papers_debated.csv"]
_DEFAULT_THEMES_OUT = "papers_themes.csv"
_DEFAULT_COMPOSITES_OUT = "papers_composite_ideas.csv"
_DEFAULT_COMPOSITES_IN = "papers_composite_ideas.csv"
_DEFAULT_SCORED_OUT = "papers_composite_scored.csv"
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
_OPENAI_TEMP = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mine emerging themes and composite startup ideas from pipeline CSVs."
    )
    p.add_argument(
        "--input",
        nargs="+",
        default=_DEFAULT_INPUTS,
        help="One or more CSV files to load (default: papers_wide_scout.csv papers_debated.csv).",
    )
    p.add_argument(
        "--min-score",
        type=int,
        default=3,
        help="Minimum overall_score to include a paper (default: 3).",
    )
    p.add_argument(
        "--keeps-only",
        action="store_true",
        help="Only include papers with claude_final_label=keep.",
    )
    p.add_argument(
        "--generate-summaries",
        action="store_true",
        help="Call OpenAI to generate a description for each theme.",
    )
    p.add_argument(
        "--generate-composites",
        action="store_true",
        help="Call OpenAI to evaluate and generate composite startup ideas.",
    )
    p.add_argument(
        "--themes-out",
        default=_DEFAULT_THEMES_OUT,
        help=f"Output path for themes CSV (default: {_DEFAULT_THEMES_OUT}).",
    )
    p.add_argument(
        "--composites-out",
        default=_DEFAULT_COMPOSITES_OUT,
        help=f"Output path for composite ideas CSV (default: {_DEFAULT_COMPOSITES_OUT}).",
    )
    p.add_argument(
        "--max-themes",
        type=int,
        default=None,
        help="Cap number of themes to process (useful for quick testing).",
    )
    p.add_argument(
        "--score-composites",
        action="store_true",
        help="Score composite ideas from --composites-in via OpenAI and write --scored-out.",
    )
    p.add_argument(
        "--composites-in",
        default=_DEFAULT_COMPOSITES_IN,
        help=f"Input path for composite ideas CSV to score (default: {_DEFAULT_COMPOSITES_IN}).",
    )
    p.add_argument(
        "--scored-out",
        default=_DEFAULT_SCORED_OUT,
        help=f"Output path for scored composite ideas CSV (default: {_DEFAULT_SCORED_OUT}).",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for LLM calls.")
    return OpenAI(api_key=api_key)


def _call_openai_json(client: OpenAI, messages: list[dict], max_tokens: int = 512) -> str:
    """Make a single OpenAI chat call and return the raw content string."""
    resp = client.chat.completions.create(
        model=_OPENAI_MODEL,
        temperature=_OPENAI_TEMP,
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=messages,
    )
    return resp.choices[0].message.content or ""


def _generate_theme_summaries(
    themes: list[Theme],
    papers_by_id: dict[str, AnalyzedPaper],
    client: OpenAI,
) -> list[Theme]:
    """Call OpenAI once per theme to generate a description and opportunities."""
    updated: list[Theme] = []
    for theme in themes:
        members = [papers_by_id[pid] for pid in theme.paper_ids if pid in papers_by_id]
        messages = build_theme_summary_prompt(theme, members)
        try:
            raw = _call_openai_json(client, messages, max_tokens=512)
            data = json.loads(raw)
            desc = data.get("description", "")
            opps = data.get("startup_opportunities", [])
            gaps = data.get("gaps", "")
            summary_parts = [desc]
            if opps:
                summary_parts.append("Opportunities: " + " | ".join(opps))
            if gaps:
                summary_parts.append("Gaps: " + gaps)
            summary = " ".join(summary_parts)
            LOGGER.info("Theme summary generated for: %s", theme.name)
        except Exception as exc:
            LOGGER.warning("Theme summary failed for %s: %s", theme.name, exc)
            summary = None
        updated.append(theme._replace(summary=summary))
    return updated


def _generate_composite_ideas(
    themes: list[Theme],
    papers_by_id: dict[str, AnalyzedPaper],
    client: OpenAI,
) -> list[CompositeIdea]:
    """Find composite candidates in each theme and call OpenAI to evaluate them."""
    ideas: list[CompositeIdea] = []
    for theme in themes:
        candidates_list = find_composite_candidates(theme, papers_by_id)
        for candidates in candidates_list:
            messages = build_composite_prompt(theme, candidates)
            paper_ids = [p.paper_id for p in candidates]
            try:
                raw = _call_openai_json(client, messages, max_tokens=768)
                idea = parse_composite_response(raw, theme.name, paper_ids)
                if idea:
                    ideas.append(idea)
                    LOGGER.info(
                        "Composite idea created: '%s' from %s", idea.title, paper_ids
                    )
            except Exception as exc:
                LOGGER.warning(
                    "Composite generation failed for %s: %s", paper_ids, exc
                )
    return ideas


def _run_score_composites(
    composites_in: str,
    scored_out: str,
    papers_by_id: dict[str, AnalyzedPaper],
    client: OpenAI,
) -> list[ScoredCompositeIdea]:
    """Load composite ideas, score each via OpenAI, and write the scored CSV.

    Args:
        composites_in: Path to the composite ideas CSV to score.
        scored_out: Path to write the scored output CSV.
        papers_by_id: Paper lookup dict for aggregating per-paper signals.
        client: Authenticated OpenAI client.

    Returns:
        List of ScoredCompositeIdea objects.
    """
    ideas = load_composite_ideas(composites_in)
    if not ideas:
        print(f"No composite ideas found in: {composites_in}")
        print("Run with --generate-composites first, or provide --composites-in.")
        return []

    print(f"\nScoring {len(ideas)} composite ideas via OpenAI...")

    def llm_fn(messages: list[dict]) -> str:
        return _call_openai_json(client, messages, max_tokens=512)

    scored = score_composite_ideas(ideas, papers_by_id, llm_fn)
    rows = scored_ideas_to_rows(scored)
    write_csv(scored_out, SCORED_COMPOSITES_CSV_COLUMNS, rows)
    print(f"Scored ideas written to: {scored_out}")
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_theme_table(themes: list[Theme]) -> None:
    print(f"\n{'THEME':<45} {'PAPERS':>6}")
    print("─" * 55)
    for t in themes:
        print(f"{t.name:<45} {len(t.paper_ids):>6}")
    total = sum(len(t.paper_ids) for t in themes)
    print("─" * 55)
    print(f"{'TOTAL':<45} {total:>6}")


def _print_composite_ideas(ideas: list[CompositeIdea]) -> None:
    if not ideas:
        print("\nNo composite ideas generated.")
        return
    print(f"\n{'─'*70}")
    print(f"COMPOSITE IDEAS ({len(ideas)} generated)")
    print(f"{'─'*70}")
    for idea in ideas:
        print(f"\n[{idea.theme_name}]")
        print(f"  Title:     {idea.title}")
        print(f"  Papers:    {', '.join(idea.paper_ids)}")
        print(f"  Core cap:  {idea.core_capability[:120]}")
        print(f"  Wedge:     {idea.wedge_description[:120]}")
        print(f"  User:      {idea.target_user[:80]}")


def _print_scored_ideas(ideas: list[ScoredCompositeIdea]) -> None:
    if not ideas:
        print("\nNo scored ideas to display.")
        return

    scored = [i for i in ideas if i.composite_score is not None]
    failed = len(ideas) - len(scored)

    # Score distribution
    if scored:
        scores = [i.composite_score for i in scored]  # type: ignore[misc]
        avg = round(sum(scores) / len(scores), 2)
        hi = max(scores)
        lo = min(scores)
        buckets = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
        for s in scores:
            bucket = min(5, max(1, round(s)))
            buckets[bucket] = buckets.get(bucket, 0) + 1

        print(f"\n{'─'*70}")
        print(f"SCORED IDEAS — {len(scored)} scored, {failed} failed")
        print(f"  avg={avg}  hi={hi}  lo={lo}")
        print(f"  Distribution: " + "  ".join(f"{k}★:{v}" for k, v in sorted(buckets.items(), reverse=True)))
        print(f"{'─'*70}")

        top5 = sorted(scored, key=lambda i: i.composite_score or 0, reverse=True)[:5]
        print("\nTOP 5 IDEAS:")
        for rank, idea in enumerate(top5, 1):
            print(f"\n  #{rank} [{idea.composite_score:.2f}] {idea.title}")
            print(f"       Theme:  {idea.theme_name}")
            print(f"       Wedge:  {idea.wedge_description[:100]}")
            scores_line = (
                f"W={idea.wedge_clarity} M={idea.market_pull} "
                f"T={idea.technical_moat} F={idea.founder_fit} S={idea.composite_synergy}"
            )
            print(f"       Scores: {scores_line}")
            if idea.scoring_notes:
                print(f"       Notes:  {idea.scoring_notes[:160]}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    # Load and filter papers
    papers = load_papers(args.input, min_score=args.min_score)
    if args.keeps_only:
        papers = [p for p in papers if p.claude_final_label == "keep"]
        LOGGER.info("keeps-only filter: %s papers remain", len(papers))

    if not papers:
        print("No papers loaded. Check --input paths and --min-score.")
        sys.exit(1)

    print(f"\nLoaded {len(papers)} papers from {args.input}")

    # Extract themes
    themes = extract_themes(papers)
    if args.max_themes:
        themes = themes[: args.max_themes]
    _print_theme_table(themes)

    papers_by_id = {p.paper_id: p for p in papers}

    # Generate theme summaries (optional LLM call)
    if args.generate_summaries:
        client = _openai_client()
        print(f"\nGenerating theme summaries via OpenAI ({len(themes)} themes)...")
        themes = _generate_theme_summaries(themes, papers_by_id, client)

    # Write themes CSV
    write_csv(args.themes_out, THEMES_CSV_COLUMNS, themes_to_rows(themes))
    print(f"\nThemes written to: {args.themes_out}")

    # Generate composite ideas (optional LLM call)
    ideas: list[CompositeIdea] = []
    if args.generate_composites:
        client = _openai_client()
        print("\nFinding composite startup ideas...")
        ideas = _generate_composite_ideas(themes, papers_by_id, client)
        _print_composite_ideas(ideas)
        if ideas:
            write_csv(args.composites_out, COMPOSITES_CSV_COLUMNS, composite_ideas_to_rows(ideas))
            print(f"\nComposite ideas written to: {args.composites_out}")
    else:
        # Still find candidates and report counts without LLM calls
        total_candidates = sum(
            len(find_composite_candidates(t, papers_by_id)) for t in themes
        )
        print(f"\nComposite candidates found (no LLM): {total_candidates}")
        print("Run with --generate-composites to evaluate and write them.")

    # Score composite ideas (optional LLM call — reads --composites-in, writes --scored-out)
    if args.score_composites:
        client = _openai_client()
        scored = _run_score_composites(args.composites_in, args.scored_out, papers_by_id, client)
        _print_scored_ideas(scored)


if __name__ == "__main__":
    main()
