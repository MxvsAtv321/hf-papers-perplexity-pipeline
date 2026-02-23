# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (Python 3.11+ required)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the pipeline
python main.py --limit 10         # Process up to 10 recent papers
python main.py --limit 5 --dry-run  # Preview without making API calls

# Run all tests
pytest

# Run a single test file
pytest tests/test_hf_feed.py
pytest tests/test_perplexity_client.py
```

## Architecture

The pipeline runs in four sequential stages orchestrated by `main.py`:

1. **Fetch** (`hf_feed.fetch_papers`) — Calls the official HF Daily Papers JSON API (`/api/daily_papers`), filters by `max_age_hours` (default 24h), and normalizes results into `Paper` dataclass instances.
2. **Deduplicate** (`notion_client.paper_already_exists`) — Queries Notion by `Paper ID` rich-text property before any API-expensive work.
3. **Enrich** (`perplexity_client.analyze_paper`) — Sends each new paper to Perplexity (`sonar-pro` by default) with a strict system prompt that demands a specific JSON schema. Has 2 retry attempts and falls back to `_extract_first_json_object` if the response isn't clean JSON.
4. **Persist** (`notion_client.create_paper_entry`) — Writes the structured analysis to Notion with exponential backoff (3 retries, handles 429s).

### Key data flow

```
HF API → [Paper] → Notion dedup check → Perplexity → Notion write
```

`models.py` defines the single shared `Paper` dataclass used across all modules. All three client modules read credentials from environment variables (no config objects passed around).

### Notion property mapping

`notion_client._build_properties` maps Perplexity JSON fields to Notion properties. Rich-text fields are truncated to 1900 chars. `product_angles` and `top_bets` lists are serialized to formatted plain-text strings before storage.

### Environment variables

| Variable | Required | Default |
|---|---|---|
| `PERPLEXITY_API_KEY` | Yes | — |
| `NOTION_API_KEY` | Yes | — |
| `NOTION_DATABASE_ID` | Yes | — |
| `PERPLEXITY_MODEL` | No | `sonar-pro` |
| `PERPLEXITY_TEMPERATURE` | No | `0.1` |

`PERPLEXITY_MODEL` and `PERPLEXITY_TEMPERATURE` are read at module import time in `perplexity_client.py`, so `.env` must be loaded before import or set in the environment.

### Tests

Tests live in `tests/` and use `conftest.py` to add the repo root to `sys.path`. Tests are unit-level only — they test internal parsing functions (`_parse_papers_payload`, `_parse_analysis_json`, `validate_analysis_schema`) without making real HTTP calls.
