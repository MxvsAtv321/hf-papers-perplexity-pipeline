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
pytest tests/test_llm_client.py
pytest tests/test_csv_sink.py
```

## Architecture

The pipeline runs in four sequential stages orchestrated by `main.py`:

1. **Fetch** (`hf_feed.fetch_papers`) — Calls the official HF Daily Papers JSON API (`/api/daily_papers`), filters by `max_age_days` (default 90, overridden by `HF_MAX_AGE_DAYS` env var), and normalizes results into `Paper` dataclass instances.
2. **Deduplicate** (`csv_sink.paper_already_exists`) — Scans the CSV output file for an existing row with the same `paper_id` before any API-expensive work.
3. **Enrich** (`llm_client.analyze_paper`) — Sends each new paper to OpenAI (`gpt-5.2` by default) with a strict system prompt that demands a specific JSON schema. Uses `response_format={"type": "json_object"}` and has 2 retry attempts.
4. **Persist** (`csv_sink.write_paper_entry`) — Appends a flattened row to a local CSV file (default `papers_pipeline.csv`).

### Key data flow

```
HF API → [Paper] → CSV dedup check → OpenAI → CSV write
```

`models.py` defines the single shared `Paper` dataclass used across all modules. All client modules read credentials from environment variables (no config objects passed around).

### CSV column mapping

`csv_sink.write_paper_entry` flattens the Paper fields and OpenAI JSON response into a single CSV row. Complex list fields (`product_angles`, `competition`, `top_bets`) are stored as JSON-serialized strings.

### Environment variables

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | Yes | — |
| `OPENAI_MODEL` | No | `gpt-5.2` |
| `OPENAI_TEMPERATURE` | No | `0.1` |
| `CSV_OUTPUT_PATH` | No | `papers_pipeline.csv` |
| `HF_MAX_AGE_DAYS` | No | `90` |

`OPENAI_MODEL` and `OPENAI_TEMPERATURE` are read at module import time in `llm_client.py`, so `.env` must be loaded before import or set in the environment.

`HF_MAX_AGE_DAYS` is read at call time inside `hf_feed.fetch_papers`, so it is safe to set it in `.env` (which `load_dotenv()` loads at the top of `main.py`).

### Tests

Tests live in `tests/` and use `conftest.py` to add the repo root to `sys.path`. Tests are unit-level only:
- `test_hf_feed.py` — tests `_parse_papers_payload` and age-filtering in `fetch_papers` (via mocked HTTP) without real HTTP calls.
- `test_llm_client.py` — tests `_parse_analysis_json` and `validate_analysis_schema` without real HTTP calls.
- `test_csv_sink.py` — tests `write_paper_entry` and `paper_already_exists` using a `tmp_path` fixture, no real file system side effects.
