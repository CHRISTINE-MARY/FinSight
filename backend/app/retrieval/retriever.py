"""Turn a user question into ranked source passages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from uuid import UUID

from app.database.documents import database_engine
from app.ingestion.embeddings import embed_texts
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.queries import (
    RankedHit,
    fetch_chunk,
    fetch_surrounding,
    full_text_search,
    load_passages,
    semantic_search,
)


@dataclass(frozen=True, slots=True)
class NeighborChunk:
    chunk_id: UUID
    chunk_index: int
    content: str


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    token_count: int
    metadata: dict[str, object]
    ticker: str
    company_name: str
    filing_type: str
    filing_date: date
    fiscal_year: int
    accession_number: str
    rrf_score: float
    semantic_rank: int | None
    lexical_rank: int | None
    neighbors: tuple[NeighborChunk, ...]


def retrieve(
    question: str, *, top_k: int = 8, candidate_k: int = 50
) -> list[RetrievedPassage]:
    """Embed the question, fuse semantic and lexical hits, and load neighbors."""
    if not question.strip():
        raise ValueError("Question is empty")
    embedding = embed_texts([question])[0]
    engine = database_engine()
    with engine.connect() as connection:
        semantic_ids = semantic_search(connection, embedding, candidate_k)
        lexical_ids = full_text_search(connection, question, candidate_k)
        if not semantic_ids and not lexical_ids:
            return []
        fused = reciprocal_rank_fusion([semantic_ids, lexical_ids])[:top_k]
        semantic_ranks = _ranks(semantic_ids)
        lexical_ranks = _ranks(lexical_ids)
        hits = [
            RankedHit(
                chunk_id=chunk_id,
                rrf_score=score,
                semantic_rank=semantic_ranks.get(chunk_id),
                lexical_rank=lexical_ranks.get(chunk_id),
            )
            for chunk_id, score in fused
        ]
        return load_passages(connection, hits)


def _ranks(chunk_ids: list[UUID]) -> dict[UUID, int]:
    ranks: dict[UUID, int] = {}
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        ranks.setdefault(chunk_id, rank)
    return ranks


class DocumentRetriever:
    """Bounded read API for hybrid search and chunk lookup."""

    def search(self, question: str, *, top_k: int = 8) -> list[RetrievedPassage]:
        return retrieve(question, top_k=top_k)

    def read_chunk(self, chunk_id: UUID) -> RetrievedPassage | None:
        engine = database_engine()
        with engine.connect() as connection:
            return fetch_chunk(connection, chunk_id)

    def read_surrounding(self, chunk_id: UUID) -> list[RetrievedPassage]:
        engine = database_engine()
        with engine.connect() as connection:
            return fetch_surrounding(connection, chunk_id)
