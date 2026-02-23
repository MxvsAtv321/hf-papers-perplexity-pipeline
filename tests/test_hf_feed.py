from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import requests

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


def _mock_resp(payload: list[dict]) -> MagicMock:
    """Return a mock requests.Response for the given payload."""
    mock = MagicMock()
    mock.json.return_value = payload
    return mock


def test_fetch_papers_age_filter_wide_window_includes_old_papers() -> None:
    """With a large max_age_days, both old and recent papers are returned."""
    now = datetime.now(UTC)
    old_dt = now - timedelta(days=200)
    recent_dt = now - timedelta(days=5)

    payload = _make_payload([
        ("2501.00001", "Old Paper", old_dt),
        ("2501.00002", "Recent Paper", recent_dt),
    ])

    with patch.dict("os.environ", {"HF_FETCH_DAYS": "1"}), \
         patch("hf_feed.requests.get", return_value=_mock_resp(payload)):
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

    with patch.dict("os.environ", {"HF_FETCH_DAYS": "1"}), \
         patch("hf_feed.requests.get", return_value=_mock_resp(payload)):
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

    with patch.dict("os.environ", {"HF_FETCH_DAYS": "1"}), \
         patch("hf_feed.requests.get", return_value=_mock_resp(payload)):
        papers = fetch_papers(max_age_days=30, limit=2)

    assert len(papers) == 2


def test_fetch_papers_multi_date_dedup() -> None:
    """Papers fetched from multiple days are deduplicated by paper_id."""
    now = datetime.now(UTC)

    day0_payload = _make_payload([
        ("2501.00001", "Paper A", now - timedelta(days=1)),
        ("2501.00002", "Paper B", now - timedelta(days=2)),
    ])
    day1_payload = _make_payload([
        ("2501.00002", "Paper B", now - timedelta(days=2)),  # duplicate
        ("2501.00003", "Paper C", now - timedelta(days=3)),
    ])

    with patch.dict("os.environ", {"HF_FETCH_DAYS": "2"}), \
         patch("hf_feed.requests.get",
               side_effect=[_mock_resp(day0_payload), _mock_resp(day1_payload)]):
        papers = fetch_papers(max_age_days=30)

    assert len(papers) == 3
    ids = {p.paper_id for p in papers}
    assert ids == {"2501.00001", "2501.00002", "2501.00003"}


def test_fetch_papers_sorted_by_published_at_desc() -> None:
    """Returned papers are ordered newest-first."""
    now = datetime.now(UTC)

    payload = _make_payload([
        ("2501.00003", "Paper C", now - timedelta(days=3)),
        ("2501.00001", "Paper A", now - timedelta(days=1)),
        ("2501.00002", "Paper B", now - timedelta(days=2)),
    ])

    with patch.dict("os.environ", {"HF_FETCH_DAYS": "1"}), \
         patch("hf_feed.requests.get", return_value=_mock_resp(payload)):
        papers = fetch_papers(max_age_days=30)

    assert papers[0].paper_id == "2501.00001"
    assert papers[1].paper_id == "2501.00002"
    assert papers[2].paper_id == "2501.00003"


def test_fetch_papers_continues_on_per_date_error() -> None:
    """A request error for one date is logged as a warning and does not abort the run."""
    now = datetime.now(UTC)

    good_payload = _make_payload([("2501.00001", "Paper A", now - timedelta(days=1))])

    with patch.dict("os.environ", {"HF_FETCH_DAYS": "2"}), \
         patch("hf_feed.requests.get",
               side_effect=[requests.RequestException("timeout"), _mock_resp(good_payload)]):
        papers = fetch_papers(max_age_days=30)

    assert len(papers) == 1
    assert papers[0].paper_id == "2501.00001"


def test_fetch_papers_hf_fetch_days_env_controls_request_count() -> None:
    """HF_FETCH_DAYS controls how many per-date requests are made."""
    now = datetime.now(UTC)
    payload = _make_payload([("2501.00001", "Paper A", now - timedelta(days=1))])

    with patch.dict("os.environ", {"HF_FETCH_DAYS": "3"}), \
         patch("hf_feed.requests.get", return_value=_mock_resp(payload)) as mock_get:
        fetch_papers(max_age_days=30)

    assert mock_get.call_count == 3
