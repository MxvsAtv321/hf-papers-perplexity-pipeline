from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from hf_feed import _parse_papers_payload, fetch_papers


def test_parse_papers_payload_smoke() -> None:
    payload = [
        {
            "paper": {
                "id": "2501.12345",
                "title": "Test Paper",
                "summary": "A short abstract.",
                "publishedAt": "2026-02-18T09:54:57.000Z",
            }
        }
    ]

    papers = _parse_papers_payload(payload)
    assert len(papers) == 1
    assert papers[0].paper_id == "2501.12345"
    assert papers[0].title == "Test Paper"
    assert papers[0].url == "https://huggingface.co/papers/2501.12345"
    assert papers[0].published_at.tzinfo == UTC


def _make_payload(papers: list[tuple[str, str, datetime]]) -> list[dict]:
    """Build a minimal HF API payload from (id, title, published_at) tuples."""
    return [
        {
            "paper": {
                "id": paper_id,
                "title": title,
                "summary": "abstract",
                "publishedAt": dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            }
        }
        for paper_id, title, dt in papers
    ]


def test_fetch_papers_age_filter_wide_window_includes_old_papers() -> None:
    """With a large max_age_days, both old and recent papers are returned."""
    now = datetime.now(UTC)
    old_dt = now - timedelta(days=200)
    recent_dt = now - timedelta(days=5)

    payload = _make_payload([
        ("2501.00001", "Old Paper", old_dt),
        ("2501.00002", "Recent Paper", recent_dt),
    ])

    mock_resp = MagicMock()
    mock_resp.json.return_value = payload

    with patch("hf_feed.requests.get", return_value=mock_resp):
        papers = fetch_papers(max_age_days=365)

    assert len(papers) == 2
    ids = {p.paper_id for p in papers}
    assert "2501.00001" in ids
    assert "2501.00002" in ids


def test_fetch_papers_age_filter_narrow_window_excludes_old_papers() -> None:
    """With a small max_age_days, only the recent paper survives filtering."""
    now = datetime.now(UTC)
    old_dt = now - timedelta(days=200)
    recent_dt = now - timedelta(days=5)

    payload = _make_payload([
        ("2501.00001", "Old Paper", old_dt),
        ("2501.00002", "Recent Paper", recent_dt),
    ])

    mock_resp = MagicMock()
    mock_resp.json.return_value = payload

    with patch("hf_feed.requests.get", return_value=mock_resp):
        papers = fetch_papers(max_age_days=10)

    assert len(papers) == 1
    assert papers[0].paper_id == "2501.00002"


def test_fetch_papers_limit_applied_after_age_filter() -> None:
    """limit= caps the final result after age filtering."""
    now = datetime.now(UTC)

    payload = _make_payload([
        ("2501.00001", "Paper A", now - timedelta(days=1)),
        ("2501.00002", "Paper B", now - timedelta(days=2)),
        ("2501.00003", "Paper C", now - timedelta(days=3)),
    ])

    mock_resp = MagicMock()
    mock_resp.json.return_value = payload

    with patch("hf_feed.requests.get", return_value=mock_resp):
        papers = fetch_papers(max_age_days=30, limit=2)

    assert len(papers) == 2
