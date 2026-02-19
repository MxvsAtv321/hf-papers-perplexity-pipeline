# Deep-Tech Idea Pipeline

Daily pipeline that:

1. Fetches recent papers from Hugging Face Daily Papers.
2. Sends each new paper to Perplexity for structured triage.
3. Saves results to a Notion database called **Deep-Tech Idea Pipeline**.

Idempotency is enforced by checking `Paper ID` in Notion before insert.

## Requirements

- Python 3.11+
- Perplexity API key
- Notion integration token + database ID

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

   - `PERPLEXITY_API_KEY`
   - `NOTION_API_KEY`
   - `NOTION_DATABASE_ID`
   - Optional: `PERPLEXITY_MODEL`, `PERPLEXITY_TEMPERATURE`

## Create Notion Database

Create a Notion database named **Deep-Tech Idea Pipeline** with these properties:

- `Paper ID` (Rich text)
- `Title` (Title)
- `URL` (URL)
- `Abstract` (Rich text)
- `Summary - Problem` (Rich text)
- `Summary - Core Method` (Rich text)
- `Summary - Key Idea` (Rich text)
- `Capability` (Rich text)
- `Product Angles` (Rich text)
- `Top Bets` (Rich text)
- `Created At` (Date)
- `Status` (Select) with option `Exploring`

Also share the database with your Notion integration/token.

### How to find `NOTION_DATABASE_ID`

- Open the database in Notion.
- From the URL, copy the 32-character ID part between the last `/` and `?v=...`.
- You can keep or remove dashes; both formats are accepted.

## Run

```bash
python main.py --limit 10
```

CLI options:

- `--limit N`: max number of recent papers to process.
- `--dry-run`: prints what would be processed, skips Perplexity and Notion writes.

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
          PERPLEXITY_API_KEY: ${{ secrets.PERPLEXITY_API_KEY }}
          NOTION_API_KEY: ${{ secrets.NOTION_API_KEY }}
          NOTION_DATABASE_ID: ${{ secrets.NOTION_DATABASE_ID }}
        run: python main.py --limit 20
```

## Testing

```bash
pytest
```
