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
    indexed_fiscal_years,
    load_passages,
    semantic_search,
)
from app.retrieval.scope import FilingScope, question_scopes, scope_query


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
# Chunks kept from each company/year search before the lists are combined.
SCOPE_BUDGET = 5


def retrieve(
    question: str,
    *,
    top_k: int = 8,
    scope_k: int = SCOPE_BUDGET,
    candidate_k: int = 50,
    prior_questions: list[str] | None = None,
    progress: Progress | None = None,
    unavailable: list[str] | None = None,
) -> list[RetrievedPassage]:
    """Search each named filing on its own, then keep that filing's top chunks.

    One company and one year still returns top_k. A comparison gives every
    company/year its own scope_k after reranking, then concatenates those
    lists and drops duplicate chunks. A fiscal year with no indexed filing
    is recorded in unavailable and is not searched.
    """
    if not question.strip():
        raise ValueError("Question is empty")
    scopes = question_scopes(question)
    gaps = unavailable if unavailable is not None else []
    engine = database_engine()
    with engine.connect() as connection:
        active, queries = _available_scopes(
            connection,
            question,
            scopes,
            prior_questions or [],
            gaps,
            progress,
        )
        if not active:
            _step(progress, "retrieve", "No corpus match")
            return []
        _step(progress, "retrieve", "Embedding the question")
        embeddings = embed_texts(queries, input_type="query")
        groups = [
            _search_scope(
                connection,
                query,
                embedding,
                scope,
                candidate_k,
                prior_questions or [],
                progress,
            )
            for scope, query, embedding in zip(
                active, queries, embeddings, strict=True
            )
        ]
        if len(groups) == 1:
            hits = groups[0][:top_k]
        else:
            hits = _combine(groups, scope_k)
        if not hits:
            _step(progress, "retrieve", "No corpus match")
            return []
        _step(progress, "retrieve", f"Reranked to {len(hits)} passages")
        _step(progress, "retrieve", "Loading passage text")
        return load_passages(connection, hits)


def _available_scopes(
    connection: Connection,
    question: str,
    scopes: list[FilingScope],
    prior_questions: list[str],
    unavailable: list[str],
    progress: Progress | None,
) -> tuple[list[FilingScope], list[str]]:
    on_record: dict[str, set[int]] = {}
    active: list[FilingScope] = []
    queries: list[str] = []
    for scope in scopes:
        if scope.ticker is not None and scope.fiscal_year is not None:
            years = on_record.get(scope.ticker)
            if years is None:
                years = indexed_fiscal_years(connection, scope.ticker)
                on_record[scope.ticker] = years
            if scope.fiscal_year not in years:
                label = f"{scope.ticker} FY{scope.fiscal_year} is not in the corpus."
                unavailable.append(label)
                _step(progress, "retrieve", label)
                continue
        active.append(scope)
        queries.append(_search_text(scope_query(question, scope), prior_questions))
    return active, queries


def _search_scope(
    connection: Connection,
    question: str,
    embedding: list[float],
    scope: FilingScope,
    candidate_k: int,
    prior_questions: list[str],
    progress: Progress | None,
) -> list[RankedHit]:
    label = _scope_label(scope)
    suffix = f" for {label}" if label else ""
    _step(progress, "retrieve", f"Running semantic search{suffix}")
    semantic_ids = semantic_search(
        connection,
        embedding,
        candidate_k,
        ticker=scope.ticker,
        fiscal_year=scope.fiscal_year,
    )
    _step(
        progress,
        "retrieve",
        f"Semantic search returned {len(semantic_ids)} chunks{suffix}",
    )
    _step(progress, "retrieve", f"Running full-text search{suffix}")
    lexical_ids = full_text_search(
        connection,
        question,
        candidate_k,
        ticker=scope.ticker,
        fiscal_year=scope.fiscal_year,
    )
    _step(
        progress,
        "retrieve",
        f"Full-text search returned {len(lexical_ids)} chunks{suffix}",
    )
    context_ids = _context_ids(
        connection, prior_questions, candidate_k, scope, progress
    )
    if not semantic_ids and not lexical_ids and not context_ids:
        return []
    _step(progress, "retrieve", f"Fusing semantic and lexical ranks{suffix}")
    semantic_ranks = _ranks(semantic_ids)
    lexical_ranks = _ranks(lexical_ids)
    return [
        RankedHit(
            chunk_id=chunk_id,
            rrf_score=score,
            semantic_rank=semantic_ranks.get(chunk_id),
            lexical_rank=lexical_ranks.get(chunk_id),
        )
        for chunk_id, score in reciprocal_rank_fusion(
            _ranked_lists(semantic_ids, lexical_ids, context_ids)
        )
    ]


def _combine(groups: list[list[RankedHit]], scope_k: int) -> list[RankedHit]:
    """Keep scope_k unique hits from each filing, in that filing's rank order."""
    chosen: list[RankedHit] = []
    seen: set[UUID] = set()
    for group in groups:
        kept = 0
        for hit in group:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            chosen.append(hit)
            kept += 1
            if kept >= scope_k:
                break
    return chosen


def _scope_label(scope: FilingScope) -> str:
    parts: list[str] = []
    if scope.ticker:
        parts.append(scope.ticker)
    if scope.fiscal_year is not None:
        parts.append(f"FY{scope.fiscal_year}")
    return " ".join(parts)


def _context_ids(
    connection: Connection,
    prior_questions: list[str],
    limit: int,
    scope: FilingScope,
    progress: Progress | None,
) -> list[UUID]:
    query = "\n".join(
        question.strip() for question in prior_questions if question.strip()
    )
    if not query:
        return []
    _step(progress, "retrieve", "Searching earlier questions")
    ids = full_text_search(
        connection,
        query,
        limit,
        ticker=scope.ticker,
        fiscal_year=scope.fiscal_year,
    )
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

    def __init__(self) -> None:
        self.unavailable: list[str] = []

    def search(
        self,
        question: str,
        *,
        top_k: int = 8,
        scope_k: int = SCOPE_BUDGET,
        prior_questions: list[str] | None = None,
        progress: Progress | None = None,
    ) -> list[RetrievedPassage]:
        self.unavailable = []
        return retrieve(
            question,
            top_k=top_k,
            scope_k=scope_k,
            prior_questions=prior_questions,
            progress=progress,
            unavailable=self.unavailable,
        )

    def read_chunk(self, chunk_id: UUID) -> RetrievedPassage | None:
        engine = database_engine()
        with engine.connect() as connection:
            return fetch_chunk(connection, chunk_id)

    def read_surrounding(self, chunk_id: UUID) -> list[RetrievedPassage]:
        engine = database_engine()
        with engine.connect() as connection:
            return fetch_surrounding(connection, chunk_id)
