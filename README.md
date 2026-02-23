# Deep-Tech Idea Pipeline

Daily pipeline that:

1. Fetches recent papers from Hugging Face Daily Papers.
2. Sends each new paper to OpenAI GPT for structured triage.
3. Saves results to a local **CSV file** (`papers_pipeline.csv` by default).

Idempotency is enforced by checking `paper_id` in the CSV before insert.

## Requirements

- Python 3.11+
- OpenAI API key

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
   - Optional: `OPENAI_MODEL` (default `gpt-5.2`)
   - Optional: `OPENAI_TEMPERATURE` (default `0.1`)
   - Optional: `CSV_OUTPUT_PATH` (default `papers_pipeline.csv`)
   - Optional: `HF_MAX_AGE_DAYS` (default `90`) — how far back to fetch HF papers

### Getting an OpenAI API key

1. Go to [platform.openai.com](https://platform.openai.com) and sign in.
2. Navigate to **API keys** in the left sidebar and click **Create new secret key**.
3. Copy the key and paste it as `OPENAI_API_KEY` in your `.env` file.

## CSV Output

Results are written to a local CSV file (default: `papers_pipeline.csv` in the project root).

Each row contains:

| Column | Description |
|---|---|
| `paper_id` | HuggingFace paper ID |
| `title` | Paper title |
| `url` | Paper URL |
| `abstract` | Paper abstract |
| `published_at` | Publication timestamp |
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
| `created_at` | Timestamp when the row was written |

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
- `--dry-run`: prints what would be processed, skips OpenAI and CSV writes.

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
        run: python main.py --limit 20
```

## Testing

```bash
pytest
```
