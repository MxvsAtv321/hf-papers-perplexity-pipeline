"""Shared typed models for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Paper:
    """Normalized paper record used across ingestion and enrichment."""

    paper_id: str
    title: str
    url: str
    abstract: str
    published_at: datetime
