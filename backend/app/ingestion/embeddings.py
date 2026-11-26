"""NVIDIA embeddings for chunk text and questions.

nvidia/nemotron-3.5-embed-1b returns a 2048-dimensional vector. Requests ask
for 1536, which is what the HNSW index can store. The model requires
input_type: passage while indexing, query while searching. A shorter vector
is L2-normalized again before it is stored or searched.
"""

from __future__ import annotations

import math
import time
from typing import Literal

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from app.config import NVIDIA_BASE_URL, settings
from app.database.models.document_chunk import EMBEDDING_DIMENSIONS

EMBEDDING_BATCH_SIZE = 8
InputType = Literal["passage", "query"]
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


def embed_texts(texts: list[str], *, input_type: InputType) -> list[list[float]]:
    if not texts:
        return []
    client = OpenAI(
        api_key=settings.nvidia_api_key,
        base_url=NVIDIA_BASE_URL,
        timeout=60.0,
        max_retries=0,
    )
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = _create_embeddings(client, batch, input_type)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(fit_stored_dimensions(list(item.embedding)) for item in ordered)
    return vectors


def _create_embeddings(client: OpenAI, batch: list[str], input_type: InputType):
    for attempt in range(_ATTEMPTS):
        try:
            return client.embeddings.create(
                model=settings.nvidia_embedding_model,
                input=batch,
                dimensions=EMBEDDING_DIMENSIONS,
                extra_body={"input_type": input_type, "truncate": "NONE"},
            )
        except (APITimeoutError, APIConnectionError):
            if attempt == _ATTEMPTS - 1:
                raise
        except APIStatusError as exc:
            if exc.status_code not in _RETRYABLE_STATUS or attempt == _ATTEMPTS - 1:
                raise
        time.sleep(2 * (attempt + 1))
    raise RuntimeError("Embedding request was not sent")
