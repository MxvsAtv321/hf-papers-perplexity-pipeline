from datetime import UTC

from hf_feed import _parse_papers_payload


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
