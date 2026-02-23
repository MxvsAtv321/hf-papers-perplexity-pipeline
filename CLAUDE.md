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
pytest tests/test_filters.py
pytest tests/test_llm_client.py
pytest tests/test_csv_sink.py
pytest tests/test_claude_agentic.py
```

## Architecture

The pipeline runs in five sequential stages orchestrated by `main.py`:

1. **Fetch** (`hf_feed.fetch_papers`) — Calls the official HF Daily Papers JSON API (`/api/daily_papers`), filters by `max_age_days` (default 180, overridden by `HF_MAX_AGE_DAYS` env var), and normalizes results into `Paper` dataclass instances.
2. **Heuristic filter** (`filters.is_potential_startup_paper`) — Keyword-based pre-filter; drops papers that are purely benchmarks, surveys, or dataset releases with no high-value signal. No LLM calls.
3. **Deduplicate** (`csv_sink.paper_already_exists`) — Scans the CSV output file for an existing row with the same `paper_id` before any API work.
4. **Score** (`llm_client.score_paper_for_startup`) — Cheap OpenAI pass: 5 integer scores (startup_potential, market_pull, technical_moat, story_for_accelerator, overall_score) + rationale. Uses `max_tokens=256`.
5. **Debate** (`claude_agentic.debate_paper_with_two_agents`) — Two-agent Claude debate triggered only when `overall_score >= AGENTIC_MIN_OVERALL_SCORE` (default 4). Agent 1 is a Technical Founder, Agent 2 is a Deep-Tech Accelerator Partner. Skipped gracefully if `ANTHROPIC_API_KEY` is not set.
6. **Enrich** (`llm_client.analyze_paper`) — Full OpenAI structured triage: summary, capability, product angles, competition, top bets.
7. **Persist** (`csv_sink.write_paper_entry`) — Appends a flattened row to a local CSV file (default `papers_pipeline.csv`).

### Key data flow

```
HF API → heuristic filter → CSV dedup → OpenAI score → Claude debate → OpenAI analysis → CSV write
```

`models.py` defines the single shared `Paper` dataclass used across all modules. All client modules read credentials from environment variables (no config objects passed around).

### Module responsibilities

| Module | Responsibility |
|---|---|
| `hf_feed.py` | Fetch and date-filter papers from the HF API |
| `filters.py` | Keyword heuristic — no LLM, no network |
| `llm_client.py` | OpenAI: startup scoring + deep analysis |
| `anthropic_client.py` | Thin wrapper around the Anthropic Messages API |
| `claude_agentic.py` | Two-agent debate logic; returns placeholder if key absent |
| `csv_sink.py` | CSV read/write; schema definition |
| `models.py` | `Paper` dataclass |
| `main.py` | Orchestrator; CLI entry point |

### CSV column mapping

`csv_sink.write_paper_entry` flattens the Paper fields, OpenAI score, Claude debate, and OpenAI analysis into a single CSV row. Complex list fields (`product_angles`, `competition`, `top_bets`) are stored as JSON-serialized strings. Score and debate columns default to empty string when not provided.

### Environment variables

| Variable | Required | Default | Read at |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | call time |
| `OPENAI_MODEL` | No | `gpt-5.2` | import time |
| `OPENAI_TEMPERATURE` | No | `0.1` | import time |
| `CSV_OUTPUT_PATH` | No | `papers_pipeline.csv` | import time |
| `HF_MAX_AGE_DAYS` | No | `180` | call time |
| `ANTHROPIC_API_KEY` | No | — | call time |
| `CLAUDE_MODEL` | No | `claude-opus-4-6` | call time |
| `AGENTIC_MIN_OVERALL_SCORE` | No | `4` | call time |

`OPENAI_MODEL` and `OPENAI_TEMPERATURE` are read at module import time in `llm_client.py`, so `.env` must be loaded before import or set in the environment. All other variables are read at call time and therefore safe to set via `load_dotenv()` at the top of `main.py`.

### Tests

Tests live in `tests/` and use `conftest.py` to add the repo root to `sys.path`. Tests are unit-level only — no real HTTP calls, no real API calls:

- `test_hf_feed.py` — tests `_parse_papers_payload` and age-filtering in `fetch_papers` (mocked HTTP).
- `test_filters.py` — parametrized tests for `is_potential_startup_paper` across 18 cases.
- `test_llm_client.py` — tests `_parse_analysis_json`, `validate_analysis_schema`, and `score_paper_for_startup` (mocked OpenAI).
- `test_csv_sink.py` — tests `write_paper_entry` and `paper_already_exists` using `tmp_path`, no real filesystem side effects.
- `test_claude_agentic.py` — tests `debate_paper_with_two_agents` (no-key placeholder, mocked Claude response, failure fallback) and `_parse_verdict` unit tests.
