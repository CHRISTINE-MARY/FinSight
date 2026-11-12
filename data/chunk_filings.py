"""Create document_chunks for every downloaded filing.

This step does not call the embedding API. Rows are stored with a null
embedding and embedding_status pending. Embed them later with
data/embed_chunks.py.

From backend/:

    uv run python ../data/chunk_filings.py
"""

from __future__ import annotations

import json
from pathlib import Path

from app.database.documents import (
    ChunkWrite,
    chunk_status_counts,
    fetch_filing,
    store_chunks,
)
from app.ingestion.chunking import document_from_html, hybrid_chunks
from app.ingestion.metadata import chunk_metadata

DATA_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = DATA_DIR / "downloads" / "manifest.json"


def _filings() -> list[dict[str, object]]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    filings = payload["filings"]
    if not isinstance(filings, list):
        raise TypeError("manifest filings must be a list")
    return filings


def chunk_filing(filing: dict[str, object]) -> None:
    accession_number = str(filing["accession_number"])
    ticker = str(filing["ticker"])
    html_path = DATA_DIR / "downloads" / Path(str(filing["local_path"]))
    document = document_from_html(html_path)
    chunks = hybrid_chunks(document)
    context = fetch_filing(accession_number)
    rows = [
        ChunkWrite(
            chunk_index=chunk.index,
            content=chunk.text,
            token_count=chunk.token_count,
            metadata=chunk_metadata(context, chunk),
        )
        for chunk in chunks
    ]
    result = store_chunks(accession_number, rows)
    print(
        f"{result.action} {ticker} {accession_number}: {result.chunk_count} chunk(s)",
        flush=True,
    )


def main() -> None:
    filings = _filings()
    print(f"Chunking {len(filings)} filing(s)", flush=True)
    for filing in filings:
        chunk_filing(filing)
    counts = chunk_status_counts()
    print(f"embedding_status counts: {counts}", flush=True)


if __name__ == "__main__":
    main()
