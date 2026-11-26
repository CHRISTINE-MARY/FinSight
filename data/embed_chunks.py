"""Embed pending document chunks, 8 texts per NVIDIA request.

A rate limit or a stalled request leaves the batch pending and retries.
Other HTTP errors mark that batch failed and the run continues.

From backend/:

    uv run python ../data/embed_chunks.py
"""

from __future__ import annotations

import time

from openai import APIConnectionError, APIStatusError, APITimeoutError
from sqlalchemy.exc import OperationalError

from app.database.documents import (
    chunk_status_counts,
    fetch_chunks_for_embedding,
    mark_embedding_failed,
    save_embeddings,
)
from app.ingestion.embeddings import EMBEDDING_BATCH_SIZE, embed_texts

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def embed_pending() -> None:
    sent = 0
    stalls = 0
    print("resuming embedding", flush=True)
    while True:
        try:
            batch = fetch_chunks_for_embedding(EMBEDDING_BATCH_SIZE)
        except OperationalError:
            stalls += 1
            _wait(stalls, "database connection failed")
            continue
        if not batch:
            break
        chunk_ids = [item.id for item in batch]
        try:
            vectors = embed_texts(
                [item.content for item in batch], input_type="passage"
            )
        except APIStatusError as exc:
            if exc.status_code in _RETRYABLE_STATUS:
                stalls += 1
                _wait(stalls, f"NVIDIA HTTP {exc.status_code}")
                continue
            if exc.status_code == 422:
                _embed_individually(batch)
                continue
            mark_embedding_failed(chunk_ids)
            print(
                f"batch marked failed after HTTP {exc.status_code}",
                flush=True,
            )
            continue
        except (APITimeoutError, APIConnectionError) as exc:
            stalls += 1
            _wait(stalls, type(exc).__name__)
            continue
        stalls = 0
        try:
            save_embeddings(list(zip(chunk_ids, vectors, strict=True)))
        except OperationalError:
            stalls += 1
            _wait(stalls, "database connection failed")
            continue
        sent += 1
        print(f"request {sent}: embedded {len(batch)} chunk(s)", flush=True)
    print(f"embedding_status counts: {chunk_status_counts()}", flush=True)


def _embed_individually(batch: list) -> None:
    """One screened passage rejects the whole batch. Save the rest one by one."""
    saved = 0
    for item in batch:
        try:
            vectors = embed_texts([item.content], input_type="passage")
        except APIStatusError as exc:
            if exc.status_code in _RETRYABLE_STATUS:
                raise
            mark_embedding_failed([item.id])
            print(
                f"chunk marked failed after HTTP {exc.status_code}",
                flush=True,
            )
            continue
        save_embeddings([(item.id, vectors[0])])
        saved += 1
    print(f"embedded {saved} chunk(s) individually", flush=True)


def _wait(stalls: int, reason: str) -> None:
    seconds = min(60, 5 * stalls)
    print(f"{reason}; batch left pending, waiting {seconds}s", flush=True)
    time.sleep(seconds)


def main() -> None:
    embed_pending()


if __name__ == "__main__":
    main()
