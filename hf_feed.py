"""Hugging Face Daily Papers ingestion helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from models import Paper

# This is an official public Hugging Face endpoint used by the Daily Papers page.
# It is preferred over HTML scraping and third-party feeds for stability.
HF_DAILY_PAPERS_API_URL = "https://huggingface.co/api/daily_papers"
REQUEST_TIMEOUT_SECONDS = 20


def fetch_papers(limit: int | None = None, max_age_hours: int | None = 24) -> list[Paper]:
    """Fetch and normalize papers from Hugging Face Daily Papers.

    Args:
        limit: Optional max number of papers to return after filtering.
        max_age_hours: Optional max age threshold in hours for recency filtering.
    """
    try:
        response = requests.get(
            HF_DAILY_PAPERS_API_URL,
            params={"limit": limit} if limit else None,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch Hugging Face papers: {exc}") from exc

    payload = response.json()
    papers = _parse_papers_payload(payload)

    if max_age_hours is not None:
        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        papers = [paper for paper in papers if paper.published_at >= cutoff]

    if limit is not None:
        papers = papers[:limit]

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
