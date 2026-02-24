# Deep-Tech Idea Pipeline

Daily pipeline that:

1. Fetches recent papers from Hugging Face Daily Papers.
2. Applies a heuristic pre-filter to drop low-signal papers (benchmarks, surveys, datasets).
3. Sends each surviving paper to OpenAI for a cheap 5-dimension startup score.
4. Runs commercially-viable high-scoring papers through a true two-agent Claude debate (Technical Founder call, then Accelerator Partner call with the founder's view).
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
   - Optional: `HF_FETCH_DAYS` (default `7`) — how many past days to fetch (~50 papers/day, deduplicated); set to `90` for ~3 months of coverage
   - Optional: `AGENTIC_MIN_OVERALL_SCORE` (default `4`) — minimum overall score to trigger debate (also requires startup_potential >= 4 OR market_pull >= 4)
   - Optional: `MAX_FETCH_DAYS_WIDE` (default `180`) — days window used in `--mode wide_scout`
   - Optional: `MAX_LLM_PAPERS_WIDE` (default `500`) — Stage 2 cap in `--mode wide_scout`
   - Optional: `MAX_DEEP_PAPERS_WIDE` (default `0`) — Stage 3-4 cap in `--mode wide_scout` (0 = skip)
   - Optional: `WIDE_SCOUT_CSV_PATH` (default `papers_wide_scout.csv`) — output file for `--mode wide_scout`

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
  │  Gate: overall_score >= AGENTIC_MIN_OVERALL_SCORE (default 4)
  │        AND (startup_potential >= 4 OR market_pull >= 4)
  │  Call 1: Technical Founder — novelty, feasibility, technical moat
  │  Call 2: Accelerator Partner — reads founder's view, gives own score + final verdict
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

### Stage tracking columns

Every row records which pipeline stages were executed for that paper.

| Column | Description |
|---|---|
| `stage1_passed` | `True` — survived date window + heuristic pre-filter (always True for any written row) |
| `stage2_scored` | `True` — OpenAI startup scoring completed |
| `stage3_debated` | `True` — Claude debate gate passed and debate was invoked; `False` if overall score or commercial signal too weak |
| `stage4_analyzed` | `True` — deep OpenAI triage completed (always True for any written row) |

The most informative flag is `stage3_debated`. A `False` there means the paper scored below the debate threshold (`overall_score < AGENTIC_MIN_OVERALL_SCORE` or both `startup_potential < 4` and `market_pull < 4`).

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

The two agents evaluate independently. Reading them side by side shows *where* they agreed and *where* they diverged.

| Column | Description |
|---|---|
| `claude_tf_score` | Technical Founder score (1–5) — weights technical novelty, feasibility, moat |
| `claude_tf_rationale` | Technical Founder 1–2 sentence rationale (≤400 chars) |
| `claude_ap_score` | Accelerator Partner score (1–5) — weights market pull, fundability, story |
| `claude_ap_rationale` | Accelerator Partner rationale (≤400 chars) |
| `claude_final_score` | Converged final score (1–5) |
| `claude_final_label` | `keep`, `maybe`, `drop`, or `unknown` (unknown = debate not configured or failed) |
| `claude_final_reason` | Final verdict reason (≤400 chars) |
| `claude_score_gap` | `\|TF score − AP score\|`; empty when either agent's score is None |
| `claude_disagreement_flag` | `True` when gap ≥ 2 **or** one agent scored ≥ 4 while the other scored ≤ 2 |
| `claude_verdict_raw` | Full JSON verdict for programmatic access |

**How to use `claude_disagreement_flag`:** Filter for `True` to find papers where the two agents saw the world differently — these are often the most interesting investment decisions, where a technical insight (TF) conflicts with a market read (AP).

**Empty Claude columns** on a row mean `stage3_debated = False` — the paper didn't pass the dual-condition debate gate and no Claude calls were made.

To configure a different output path:

```bash
CSV_OUTPUT_PATH=/path/to/my_output.csv
```

To open in Excel or Google Sheets, just open the CSV file directly — both apps handle UTF-8 CSV natively.

## Run

### Full mode (default)

Fetches the past `HF_FETCH_DAYS` days of HF papers, runs all four pipeline stages, and writes results to `papers_pipeline.csv`.

```bash
python main.py --limit 10          # process up to 10 papers
python main.py --limit 5 --dry-run # preview without API calls or CSV writes
```

### Wide scout mode

Broad-coverage sweep that trades depth for breadth. Useful for discovering papers across months of history without spending on Stage 3-4 for every paper.

```bash
python main.py --mode wide_scout
python main.py --mode wide_scout --dry-run   # preview: shows how many papers would be scored
```

Behavior:
- Fetches `MAX_FETCH_DAYS_WIDE` (default 180) days of HF papers — up to ~9,000 papers.
- Applies the heuristic filter and deduplicates against `papers_wide_scout.csv`.
- Runs Stage 2 (OpenAI startup scoring) on up to `MAX_LLM_PAPERS_WIDE` (default 500) papers.
- If `MAX_DEEP_PAPERS_WIDE > 0`, the top-N papers by `overall_score` also get Stage 3 (Claude debate) and Stage 4 (deep analysis).
- Results are written to `papers_wide_scout.csv` — completely separate from `papers_pipeline.csv`.
- Idempotent: re-running skips papers already in `papers_wide_scout.csv`.

Example — score 500 papers, deep-dive top 20:

```bash
MAX_DEEP_PAPERS_WIDE=20 python main.py --mode wide_scout
```

CLI options:

- `--limit N`: max number of recent papers to process (**full mode only**).
- `--mode {full,wide_scout}`: pipeline mode (default: `full`).
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
