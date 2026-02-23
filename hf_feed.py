"""Hugging Face Daily Papers ingestion helpers."""

from __future__ import annotations

import logging
import os
from datetime import UTC, date, datetime, timedelta
from typing import Any

import requests

from models import Paper

# This is an official public Hugging Face endpoint used by the Daily Papers page.
# It is preferred over HTML scraping and third-party feeds for stability.
HF_DAILY_PAPERS_API_URL = "https://huggingface.co/api/daily_papers"
REQUEST_TIMEOUT_SECONDS = 20
_DEFAULT_MAX_AGE_DAYS = 180
_DEFAULT_FETCH_DAYS = 7


def fetch_papers(limit: int | None = None, max_age_days: int | None = None) -> list[Paper]:
    """Fetch and normalize papers from Hugging Face Daily Papers.

    Fetches from the past HF_FETCH_DAYS days (default 7) by iterating over
    ?date=YYYY-MM-DD query parameters, then deduplicates, age-filters, and
    applies the limit.

    Args:
        limit: Optional max number of papers to return after age filtering.
        max_age_days: Max age in days for recency filtering. Reads HF_MAX_AGE_DAYS
            env var if not supplied; defaults to 180.
    """
    if max_age_days is None:
        max_age_days = int(os.environ.get("HF_MAX_AGE_DAYS", _DEFAULT_MAX_AGE_DAYS))

    fetch_days = int(os.environ.get("HF_FETCH_DAYS", _DEFAULT_FETCH_DAYS))
    today = datetime.now(UTC).date()

    papers_by_id: dict[str, Paper] = {}

    for days_ago in range(fetch_days):
        fetch_date: date = today - timedelta(days=days_ago)
        url = f"{HF_DAILY_PAPERS_API_URL}?date={fetch_date.isoformat()}"

        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exc:
            logging.warning(
                "HF fetch: failed for date=%s, skipping: %s", fetch_date, exc
            )
            continue

        day_papers = _parse_papers_payload(response.json())
        new_ids = 0
        for paper in day_papers:
            if paper.paper_id not in papers_by_id:
                papers_by_id[paper.paper_id] = paper
                new_ids += 1

        logging.info(
            "HF fetch: date=%s raw_count=%s new_unique=%s",
            fetch_date,
            len(day_papers),
            new_ids,
        )

    papers = list(papers_by_id.values())
    raw_count = len(papers)

    cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
    papers = [paper for paper in papers if paper.published_at >= cutoff]
    filtered_count = len(papers)

    papers.sort(key=lambda p: p.published_at, reverse=True)

    if limit is not None:
        papers = papers[:limit]

    logging.info(
        "HF fetch: raw_count=%s, after_age_filter=%s, max_age_days=%s, "
        "fetch_days=%s, limit=%s, returned=%s",
        raw_count,
        filtered_count,
        max_age_days,
        fetch_days,
        limit,
        len(papers),
    )

    return papers


def _parse_papers_payload(payload: Any) -> list[Paper]:
    """Parse API payload into normalized Paper objects."""
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Daily Papers payload shape: expected a list")

    parsed: list[Paper] = []
    for item in payload:
        if not isinstance(item, dict):
            continue

        paper_block = item.get("paper") if isinstance(item.get("paper"), dict) else {}
        paper_id = _as_str(paper_block.get("id")) or _as_str(item.get("id"))
        title = _as_str(paper_block.get("title")) or _as_str(item.get("title"))
        summary = _as_str(paper_block.get("summary")) or _as_str(item.get("summary"))
        published_raw = _as_str(paper_block.get("publishedAt")) or _as_str(item.get("publishedAt"))

        if not paper_id or not title:
            continue

        parsed.append(
            Paper(
                paper_id=paper_id,
                title=title,
                url=f"https://huggingface.co/papers/{paper_id}",
                abstract=summary,
                published_at=_parse_datetime_or_now(published_raw),
            )
        )

    return parsed


def _parse_datetime_or_now(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(UTC)

    # HF API typically returns RFC3339 timestamps with trailing Z.
    value = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(UTC)

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _as_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
