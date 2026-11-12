"""OpenRouter embeddings for chunk text.

nvidia/nemotron-3.5-embed-1b:free returns a 2048-dimensional vector.
document_chunks.embedding is vector(1536), which is what the HNSW index can
store. This model supports shorter vectors: keep the leading dimensions and
L2-normalize again.
"""

from __future__ import annotations

import math
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from app.config import settings
from app.database.models.document_chunk import EMBEDDING_DIMENSIONS

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_BATCH_SIZE = 8
_ATTEMPTS = 3
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def fit_stored_dimensions(vector: list[float]) -> list[float]:
    """Keep the leading stored dimensions and L2-normalize."""
    if len(vector) < EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Expected at least {EMBEDDING_DIMENSIONS} dimensions, got {len(vector)}"
        )
    trimmed = vector[:EMBEDDING_DIMENSIONS]
    norm = math.sqrt(sum(value * value for value in trimmed))
    if norm == 0:
        raise RuntimeError("Embedding norm is zero")
    return [value / norm for value in trimmed]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "http://localhost:5173",
            "X-Title": "Document Copilot",
        },
        timeout=70.0,
        max_retries=0,
    )
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = _create_embeddings(client, batch)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(fit_stored_dimensions(list(item.embedding)) for item in ordered)
    return vectors


def _create_embeddings(client: OpenAI, batch: list[str]):
    for attempt in range(_ATTEMPTS):
        try:
            return client.embeddings.create(
                model=settings.openrouter_embedding_model,
                input=batch,
            )
        except (APITimeoutError, APIConnectionError):
            if attempt == _ATTEMPTS - 1:
                raise
        except APIStatusError as exc:
            if exc.status_code not in _RETRYABLE_STATUS or attempt == _ATTEMPTS - 1:
                raise
        time.sleep(2 * (attempt + 1))
    raise RuntimeError("Embedding request was not sent")
