"""Chunking and embedding are separate commands.

From backend/:

    uv run python ../data/chunk_filings.py
    uv run python ../data/embed_chunks.py
"""

raise SystemExit(
    "Use data/chunk_filings.py to store chunks, then data/embed_chunks.py to embed them."
)
