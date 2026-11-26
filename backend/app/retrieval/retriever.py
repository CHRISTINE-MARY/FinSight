"""Turn a user question into ranked source passages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from uuid import UUID

from sqlalchemy.engine import Connection

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


Progress = Callable[[str, str], None]


def retrieve(
    question: str,
    *,
    top_k: int = 8,
    candidate_k: int = 50,
    prior_questions: list[str] | None = None,
    progress: Progress | None = None,
) -> list[RetrievedPassage]:
    """Embed the question, fuse semantic and lexical hits, and load neighbors.

    Earlier user questions are repeated less often than the current question
    in the embedded text, and they add one full-text list. A follow-up that
    omits the company still searches that company, without a second embedding.
    """
    if not question.strip():
        raise ValueError("Question is empty")
    _step(progress, "retrieve", "Embedding the question")
    embedding = embed_texts(
        [_search_text(question, prior_questions or [])], input_type="query"
    )[0]
    engine = database_engine()
    with engine.connect() as connection:
        _step(progress, "retrieve", "Running semantic search")
        semantic_ids = semantic_search(connection, embedding, candidate_k)
        _step(
            progress,
            "retrieve",
            f"Semantic search returned {len(semantic_ids)} chunks",
        )
        _step(progress, "retrieve", "Running full-text search")
        lexical_ids = full_text_search(connection, question, candidate_k)
        _step(
            progress,
            "retrieve",
            f"Full-text search returned {len(lexical_ids)} chunks",
        )
        context_ids = _context_ids(
            connection, prior_questions or [], candidate_k, progress
        )
        if not semantic_ids and not lexical_ids and not context_ids:
            _step(progress, "retrieve", "No corpus match")
            return []
        _step(progress, "retrieve", "Fusing semantic and lexical ranks")
        fused = reciprocal_rank_fusion(
            _ranked_lists(semantic_ids, lexical_ids, context_ids)
        )[:top_k]
        _step(progress, "retrieve", f"Fused to {len(fused)} passages")
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
        _step(progress, "retrieve", "Loading passage text")
        return load_passages(connection, hits)


def _context_ids(
    connection: Connection,
    prior_questions: list[str],
    limit: int,
    progress: Progress | None,
) -> list[UUID]:
    query = "\n".join(
        question.strip() for question in prior_questions if question.strip()
    )
    if not query:
        return []
    _step(progress, "retrieve", "Searching earlier questions")
    ids = full_text_search(connection, query, limit)
    noun = "chunk" if len(ids) == 1 else "chunks"
    _step(progress, "retrieve", f"Earlier questions returned {len(ids)} {noun}")
    return ids


def _search_text(question: str, prior_questions: list[str]) -> str:
    """Repeat the current question so it outweighs the earlier ones."""
    earlier = [item.strip() for item in prior_questions if item.strip()]
    if not earlier:
        return question
    return "\n".join([question, question, *earlier])


def _ranked_lists(
    semantic_ids: list[UUID],
    lexical_ids: list[UUID],
    context_ids: list[UUID],
) -> list[list[UUID]]:
    if not context_ids:
        return [semantic_ids, lexical_ids]
    return [semantic_ids, lexical_ids, context_ids]


def _step(progress: Progress | None, phase: str, detail: str) -> None:
    if progress is not None:
        progress(phase, detail)


def _ranks(chunk_ids: list[UUID]) -> dict[UUID, int]:
    ranks: dict[UUID, int] = {}
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        ranks.setdefault(chunk_id, rank)
    return ranks


class DocumentRetriever:
    """Bounded read API for hybrid search and chunk lookup."""

    def search(
        self,
        question: str,
        *,
        top_k: int = 8,
        prior_questions: list[str] | None = None,
        progress: Progress | None = None,
    ) -> list[RetrievedPassage]:
        return retrieve(
            question,
            top_k=top_k,
            prior_questions=prior_questions,
            progress=progress,
        )

    def read_chunk(self, chunk_id: UUID) -> RetrievedPassage | None:
        engine = database_engine()
        with engine.connect() as connection:
            return fetch_chunk(connection, chunk_id)

    def read_surrounding(self, chunk_id: UUID) -> list[RetrievedPassage]:
        engine = database_engine()
        with engine.connect() as connection:
            return fetch_surrounding(connection, chunk_id)
