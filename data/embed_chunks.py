"""Embed pending document chunks, 8 texts per OpenRouter request.

Stops when the free-model daily request cap is spent. A later run continues
from the chunks still marked pending.

From backend/:

    uv run python ../data/embed_chunks.py
"""

from __future__ import annotations

import httpx
from openai import APIStatusError

from app.config import settings
from app.database.documents import (
    chunk_status_counts,
    fetch_chunks_for_embedding,
    mark_embedding_failed,
    save_embeddings,
)
from app.ingestion.embeddings import EMBEDDING_BATCH_SIZE, embed_texts


def free_requests_remaining() -> int:
    """How many free-model requests this key can still make today."""
    response = httpx.get(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        timeout=20.0,
    )
    response.raise_for_status()
    quota = response.json()["data"]["free_model_daily_requests"]
    remaining = int(quota["remaining"])
    print(
        f"free requests remaining: {remaining} of {quota['limit']}",
        flush=True,
    )
    return remaining


def embed_pending(max_requests: int) -> None:
    sent = 0
    while sent < max_requests:
        batch = fetch_chunks_for_embedding(EMBEDDING_BATCH_SIZE)
        if not batch:
            break
        chunk_ids = [item.id for item in batch]
        try:
            vectors = embed_texts([item.content for item in batch])
        except APIStatusError as exc:
            if exc.status_code == 429:
                print("free request limit reached; batch left pending", flush=True)
                break
            mark_embedding_failed(chunk_ids)
            raise
        except Exception:
            mark_embedding_failed(chunk_ids)
            raise
        save_embeddings(list(zip(chunk_ids, vectors, strict=True)))
        sent += 1
        print(
            f"request {sent}/{max_requests}: embedded {len(batch)} chunk(s)",
            flush=True,
        )
    print(f"embedding_status counts: {chunk_status_counts()}", flush=True)


def main() -> None:
    embed_pending(free_requests_remaining())


if __name__ == "__main__":
    main()
