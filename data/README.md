# Data

Local data artifacts for development live here.

- `downloads/` holds raw source files fetched from SEC EDGAR, grouped by year.
- `markdown/` mirrors `downloads/` with each filing converted to Markdown.
- Downloaded payloads and the Markdown copies are gitignored because the corpus can get large.
- Fetch a sample corpus with `uv run data/download.py`.
- Load the Markdown filings into `source_documents` from `backend/` with `uv run python ../data/load_source_documents.py`.
- Create chunks for every filing, without calling the embedding API, from `backend/` with `uv run python ../data/chunk_filings.py`.
- Embed pending chunks, 8 per request, until the free daily request cap is spent, from `backend/` with `uv run python ../data/embed_chunks.py`. 
