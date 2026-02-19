# Deep-Tech Idea Pipeline Architecture

`main.py` orchestrates a semi-autonomous daily run in four simple stages:

1. **Ingest papers** from Hugging Face Daily Papers (`hf_feed.fetch_papers`).
2. **Enrich each new paper** via Perplexity (`perplexity_client.analyze_paper`).
3. **Prevent duplicates** by checking Notion for `Paper ID` (`notion_client.paper_already_exists`).
4. **Persist analysis** to the Notion database (`notion_client.create_paper_entry`).

Idempotency is achieved by making `Paper ID` the dedupe key and querying Notion before inserts.
