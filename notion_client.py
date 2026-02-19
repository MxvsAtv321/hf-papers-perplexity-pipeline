"""Notion API integration for the Deep-Tech Idea Pipeline database."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from typing import Any

import requests

from models import Paper

NOTION_API_BASE_URL = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3


def paper_already_exists(paper_id: str) -> bool:
    """Check whether a page with the given Paper ID already exists in Notion."""
    database_id, headers = _notion_context()
    payload = {
        "filter": {
            "property": "Paper ID",
            "rich_text": {"equals": paper_id},
        },
        "page_size": 1,
    }

    response = _request_with_backoff(
        method="POST",
        url=f"{NOTION_API_BASE_URL}/databases/{database_id}/query",
        headers=headers,
        json_payload=payload,
    )
    body = response.json()
    results = body.get("results", [])
    return bool(results)


def create_paper_entry(paper: Paper, analysis: dict[str, Any]) -> None:
    """Create a Notion page for one analyzed paper."""
    database_id, headers = _notion_context()
    properties = _build_properties(paper=paper, analysis=analysis)
    payload = {"parent": {"database_id": database_id}, "properties": properties}

    _request_with_backoff(
        method="POST",
        url=f"{NOTION_API_BASE_URL}/pages",
        headers=headers,
        json_payload=payload,
    )


def _build_properties(paper: Paper, analysis: dict[str, Any]) -> dict[str, Any]:
    summary = analysis.get("summary", {}) if isinstance(analysis.get("summary"), dict) else {}
    capability = analysis.get("capability", {}) if isinstance(analysis.get("capability"), dict) else {}

    return {
        "Paper ID": _rich_text_property(paper.paper_id),
        "Title": _title_property(paper.title),
        "URL": {"url": paper.url},
        "Abstract": _rich_text_property(paper.abstract),
        "Summary - Problem": _rich_text_property(_as_text(summary.get("problem"))),
        "Summary - Core Method": _rich_text_property(_as_text(summary.get("core_method"))),
        "Summary - Key Idea": _rich_text_property(_as_text(summary.get("key_technical_idea"))),
        "Capability": _rich_text_property(_as_text(capability.get("plain_language_capability"))),
        "Product Angles": _rich_text_property(_format_product_angles(analysis.get("product_angles"))),
        "Top Bets": _rich_text_property(_format_top_bets(analysis.get("top_bets"))),
        "Created At": {"date": {"start": datetime.now(UTC).date().isoformat()}},
        "Status": {"select": {"name": "Exploring"}},
    }


def _format_product_angles(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    lines: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = _as_text(item.get("name")) or "Unnamed angle"
        persona = _as_text(item.get("target_user_persona"))
        statement = _as_text(item.get("problem_statement"))
        ten_x = _as_text(item.get("ten_x_improvement"))
        lines.append(f"- {name}")
        if persona:
            lines.append(f"  Persona: {persona}")
        if statement:
            lines.append(f"  Problem: {statement}")
        if ten_x:
            lines.append(f"  10x: {ten_x}")
    return "\n".join(lines)


def _format_top_bets(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    lines: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        angle = _as_text(item.get("product_angle_name")) or "Unknown angle"
        rationale = _as_text(item.get("rationale"))
        lines.append(f"- {angle}: {rationale}")
    return "\n".join(lines)


def _title_property(text: str) -> dict[str, Any]:
    return {"title": [{"text": {"content": _truncate(text)}}]}


def _rich_text_property(text: str) -> dict[str, Any]:
    if not text:
        return {"rich_text": []}
    return {"rich_text": [{"text": {"content": _truncate(text)}}]}


def _truncate(text: str, max_len: int = 1900) -> str:
    value = text.strip()
    return value if len(value) <= max_len else f"{value[: max_len - 3]}..."


def _as_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _notion_context() -> tuple[str, dict[str, str]]:
    api_key = os.getenv("NOTION_API_KEY")
    database_id = os.getenv("NOTION_DATABASE_ID")
    if not api_key:
        raise RuntimeError("NOTION_API_KEY environment variable is required")
    if not database_id:
        raise RuntimeError("NOTION_DATABASE_ID environment variable is required")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    return database_id, headers


def _request_with_backoff(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    json_payload: dict[str, Any],
) -> requests.Response:
    """Send a Notion request with simple exponential backoff for rate limits."""
    delay_seconds = 1.0
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=json_payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 429 and attempt < MAX_RETRIES:
                time.sleep(delay_seconds)
                delay_seconds *= 2
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break
            time.sleep(delay_seconds)
            delay_seconds *= 2

    response_text = ""
    if isinstance(last_error, requests.HTTPError) and last_error.response is not None:
        try:
            response_text = json.dumps(last_error.response.json())
        except ValueError:
            response_text = last_error.response.text

    raise RuntimeError(f"Notion API request failed after retries: {last_error} {response_text}")
