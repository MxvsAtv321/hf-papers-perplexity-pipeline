# Deep-Tech Idea Pipeline

Daily pipeline that:

1. Fetches recent papers from Hugging Face Daily Papers.
2. Applies a heuristic pre-filter to drop low-signal papers (benchmarks, surveys, datasets).
3. Sends each surviving paper to OpenAI for a cheap 5-dimension startup score.
4. Runs high-scoring papers through a two-agent Claude debate (Technical Founder vs. Accelerator Partner).
5. Saves full results to a local **CSV file** (`papers_pipeline.csv` by default).

Idempotency is enforced by checking `paper_id` in the CSV before any API work.

## Requirements

- Python 3.11+
- OpenAI API key (required)
- Anthropic API key (optional — debate stage is skipped gracefully if absent)

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create `.env` from `.env.example`:

   ```bash
   cp .env.example .env
   ```

4. Fill values in `.env`:

   - `OPENAI_API_KEY` — your OpenAI API key (required)
   - `ANTHROPIC_API_KEY` — your Anthropic API key (optional; debate stage skipped if blank)
   - Optional: `OPENAI_MODEL` (default `gpt-5.2`)
   - Optional: `OPENAI_TEMPERATURE` (default `0.1`)
   - Optional: `CLAUDE_MODEL` (default `claude-opus-4-6`)
   - Optional: `CSV_OUTPUT_PATH` (default `papers_pipeline.csv`)
   - Optional: `HF_MAX_AGE_DAYS` (default `180`) — how far back to fetch HF papers
   - Optional: `AGENTIC_MIN_OVERALL_SCORE` (default `4`) — minimum OpenAI score to trigger debate

### Getting API keys

**OpenAI:**
1. Go to [platform.openai.com](https://platform.openai.com) and sign in.
2. Navigate to **API keys** and click **Create new secret key**.
3. Paste as `OPENAI_API_KEY` in your `.env`.

**Anthropic (optional):**
1. Go to [console.anthropic.com](https://console.anthropic.com) and sign in.
2. Navigate to **API keys** and create a new key.
3. Paste as `ANTHROPIC_API_KEY` in your `.env`.

## Pipeline stages

```
HF API
  │
  ▼
Stage 1 — Heuristic filter (filters.py)
  │  Drops benchmarks, surveys, dataset-only papers — no LLM calls
  ▼
CSV dedup check (paper_already_exists)
  │  Skips papers already in the output CSV
  ▼
Stage 2 — OpenAI startup scoring (llm_client.score_paper_for_startup)
  │  Cheap pass: 5 integer scores + rationale (max_tokens=256)
  │  Model: gpt-5.2 (configurable via OPENAI_MODEL)
  ▼
Stage 3 — Claude two-agent debate (claude_agentic.debate_paper_with_two_agents)
  │  Triggered only when overall_score >= AGENTIC_MIN_OVERALL_SCORE
  │  Agent 1: Technical Founder — novelty, feasibility, technical moat
  │  Agent 2: Deep-Tech Accelerator Partner — market pull, fundability
  │  Skipped gracefully if ANTHROPIC_API_KEY is not set
  ▼
Stage 4 — OpenAI deep analysis (llm_client.analyze_paper)
  │  Full structured triage: summary, capability, product angles, competition, top bets
  ▼
CSV write (csv_sink.write_paper_entry)
```

## CSV Output

Results are written to a local CSV file (default: `papers_pipeline.csv` in the project root).

### Core columns

| Column | Description |
|---|---|
| `paper_id` | HuggingFace paper ID |
| `title` | Paper title |
| `url` | Paper URL |
| `abstract` | Paper abstract |
| `published_at` | Publication timestamp |
| `created_at` | Timestamp when the row was written |

### Stage 4 — Deep analysis columns

| Column | Description |
|---|---|
| `summary_problem` | Problem the paper addresses |
| `summary_core_method` | Core method used |
| `summary_key_technical_idea` | Key technical idea |
| `summary_inputs_outputs` | Inputs and outputs |
| `summary_data_assumptions` | Data and assumptions |
| `summary_metrics_and_baselines` | Metrics and baselines |
| `summary_limitations` | Limitations |
| `capability_plain_language_capability` | Plain-language capability description |
| `product_angles` | JSON array of product angle objects |
| `competition` | JSON array of competitor objects |
| `top_bets` | JSON array of top bet objects |

### Stage 2 — OpenAI startup score columns

| Column | Description |
|---|---|
| `startup_potential` | Score 1–5: overall startup viability |
| `market_pull` | Score 1–5: strength of market demand |
| `technical_moat` | Score 1–5: defensibility of technical advantage |
| `story_for_accelerator` | Score 1–5: pitch quality for accelerators |
| `overall_score` | Score 1–5: aggregated OpenAI verdict |
| `score_rationale` | Short rationale from OpenAI |

### Stage 3 — Claude debate columns

| Column | Description |
|---|---|
| `claude_technical_founder_score` | Technical Founder agent score (1–5) |
| `claude_technical_founder_rationale` | Technical Founder rationale |
| `claude_accelerator_partner_score` | Accelerator Partner agent score (1–5) |
| `claude_accelerator_partner_rationale` | Accelerator Partner rationale |
| `claude_final_score` | Converged final score (1–5) |
| `claude_final_label` | `keep`, `maybe`, or `drop` |
| `claude_final_reason` | Final verdict reason |

To configure a different output path:

```bash
CSV_OUTPUT_PATH=/path/to/my_output.csv
```

To open in Excel or Google Sheets, just open the CSV file directly — both apps handle UTF-8 CSV natively.

## Run

```bash
python main.py --limit 10
```

CLI options:

- `--limit N`: max number of recent papers to process.
- `--dry-run`: prints what would be processed, skips all API calls and CSV writes.

## Scheduling

### Cron example

```cron
0 7 * * * /usr/bin/python /path/to/repo/main.py --limit 20 >> /var/log/hf_papers.log 2>&1
```

### GitHub Actions example

Create `.github/workflows/daily_pipeline.yml`:

```yaml
name: Daily Deep-Tech Pipeline

on:
  schedule:
    - cron: "0 7 * * *"
  workflow_dispatch:

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install deps
        run: pip install -r requirements.txt
      - name: Run pipeline
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: python main.py --limit 20
```

## Testing

```bash
pytest
```
