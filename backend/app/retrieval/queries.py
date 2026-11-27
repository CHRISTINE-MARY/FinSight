"""pgvector and Postgres full-text queries over document_chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.engine import Connection, Result

if TYPE_CHECKING:
    from app.retrieval.retriever import RetrievedPassage

# <=> is cosine distance and matches the HNSW index (vector_cosine_ops).
# The vector type lives in the extensions schema on Supabase.
SEMANTIC_SEARCH = text(
    """
    SELECT chunk.id
    FROM document_chunks AS chunk
    JOIN source_documents AS source ON source.id = chunk.document_id
    WHERE chunk.embedding_status = 'embedded'
      AND chunk.embedding IS NOT NULL
      AND (CAST(:ticker AS text) IS NULL OR source.ticker = :ticker)
      AND (CAST(:fiscal_year AS integer) IS NULL OR source.fiscal_year = :fiscal_year)
    ORDER BY chunk.embedding <=> CAST(:query_vector AS extensions.vector)
    LIMIT :limit
    """
)

# plainto_tsquery ANDs every word, so a paragraph-length analyst question
# matches nothing. OR the English lexemes instead, the same job BM25 does:
# a filing that shares a few distinctive terms still ranks.
FULL_TEXT_SEARCH = text(
    """
    WITH lexemes AS (
        SELECT lexeme
        FROM unnest(to_tsvector('english', :query)) AS token(lexeme, positions)
        WHERE lexeme ~ '^[a-z0-9]+$'
    ),
    query AS (
        SELECT to_tsquery('english', string_agg(lexeme, ' | ')) AS tsq
        FROM lexemes
    )
    SELECT chunk.id
    FROM document_chunks AS chunk
    JOIN source_documents AS source ON source.id = chunk.document_id
    CROSS JOIN query
    WHERE query.tsq IS NOT NULL
      AND chunk.search_vector @@ query.tsq
      AND (CAST(:ticker AS text) IS NULL OR source.ticker = :ticker)
      AND (CAST(:fiscal_year AS integer) IS NULL OR source.fiscal_year = :fiscal_year)
    ORDER BY ts_rank_cd(chunk.search_vector, query.tsq) DESC
    LIMIT :limit
    """
)

# Neighbors are the chunks immediately before and after each hit so grounding
# can show the surrounding passage without treating them as extra search hits.
LOAD_PASSAGES = text(
    """
    WITH hits AS (
        SELECT id, document_id, chunk_index
        FROM document_chunks
        WHERE id = ANY(:chunk_ids)
    )
    SELECT
        hits.id AS hit_id,
        chunk.id AS id,
        chunk.document_id,
        chunk.chunk_index,
        chunk.content,
        chunk.token_count,
        chunk.metadata,
        source.ticker,
        source.company_name,
        source.filing_type,
        source.filing_date,
        source.fiscal_year,
        source.accession_number
    FROM hits
    JOIN document_chunks AS chunk
        ON chunk.document_id = hits.document_id
       AND chunk.chunk_index BETWEEN hits.chunk_index - 1 AND hits.chunk_index + 1
    JOIN source_documents AS source
        ON source.id = chunk.document_id
    """
).bindparams(bindparam("chunk_ids", type_=ARRAY(PG_UUID(as_uuid=True))))

_CHUNK_COLUMNS = """
    chunk.id AS id,
    chunk.document_id,
    chunk.chunk_index,
    chunk.content,
    chunk.token_count,
    chunk.metadata,
    source.ticker,
    source.company_name,
    source.filing_type,
    source.filing_date,
    source.fiscal_year,
    source.accession_number
"""

FETCH_CHUNK = text(
    f"""
    SELECT {_CHUNK_COLUMNS}
    FROM document_chunks AS chunk
    JOIN source_documents AS source ON source.id = chunk.document_id
    WHERE chunk.id = :chunk_id
    """
)

FETCH_SURROUNDING = text(
    f"""
    WITH center AS (
        SELECT document_id, chunk_index
        FROM document_chunks
        WHERE id = :chunk_id
    )
    SELECT {_CHUNK_COLUMNS}
    FROM document_chunks AS chunk
    JOIN center
        ON chunk.document_id = center.document_id
       AND chunk.chunk_index BETWEEN center.chunk_index - 1 AND center.chunk_index + 1
       AND chunk.id <> :chunk_id
    JOIN source_documents AS source ON source.id = chunk.document_id
    ORDER BY chunk.chunk_index
    """
)


@dataclass(frozen=True, slots=True)
class RankedHit:
    chunk_id: UUID
    rrf_score: float
    semantic_rank: int | None
    lexical_rank: int | None


INDEXED_YEARS = text(
    """
    SELECT DISTINCT fiscal_year
    FROM source_documents
    WHERE ticker = :ticker
    """
)


def indexed_fiscal_years(connection: Connection, ticker: str) -> set[int]:
    """Fiscal years that already have a filing stored for this company."""
    rows = connection.execute(INDEXED_YEARS, {"ticker": ticker}).all()
    years: set[int] = set()
    for row in rows:
        value = row[0]
        if isinstance(value, int):
            years.add(value)
    return years


def semantic_search(
    connection: Connection,
    query_embedding: list[float],
    limit: int,
    *,
    ticker: str | None = None,
    fiscal_year: int | None = None,
) -> list[UUID]:
    rows = connection.execute(
        SEMANTIC_SEARCH,
        {
            "query_vector": _vector_literal(query_embedding),
            "limit": limit,
            "ticker": ticker,
            "fiscal_year": fiscal_year,
        },
    )
    return _ids(rows)


def full_text_search(
    connection: Connection,
    query: str,
    limit: int,
    *,
    ticker: str | None = None,
    fiscal_year: int | None = None,
) -> list[UUID]:
    rows = connection.execute(
        FULL_TEXT_SEARCH,
        {
            "query": query,
            "limit": limit,
            "ticker": ticker,
            "fiscal_year": fiscal_year,
        },
    )
    return _ids(rows)


def load_passages(
    connection: Connection, hits: list[RankedHit]
) -> list[RetrievedPassage]:
    from app.retrieval.retriever import NeighborChunk, RetrievedPassage

    if not hits:
        return []
    rows = connection.execute(
        LOAD_PASSAGES, {"chunk_ids": [hit.chunk_id for hit in hits]}
    ).all()
    grouped: dict[UUID, list[object]] = {}
    for row in rows:
        grouped.setdefault(_uuid(row.hit_id), []).append(row)

    passages: list[RetrievedPassage] = []
    for hit in hits:
        group = grouped.get(hit.chunk_id, [])
        center = next((row for row in group if _uuid(row.id) == hit.chunk_id), None)
        if center is None:
            raise LookupError(f"Chunk {hit.chunk_id} was not found")
        neighbors = tuple(
            NeighborChunk(
                chunk_id=_uuid(row.id),
                chunk_index=row.chunk_index,
                content=row.content,
            )
            for row in sorted(group, key=lambda row: row.chunk_index)
            if _uuid(row.id) != hit.chunk_id
        )
        passages.append(
            RetrievedPassage(
                chunk_id=hit.chunk_id,
                document_id=_uuid(center.document_id),
                chunk_index=center.chunk_index,
                content=center.content,
                token_count=center.token_count,
                metadata=dict(center.metadata or {}),
                ticker=center.ticker,
                company_name=center.company_name,
                filing_type=center.filing_type,
                filing_date=center.filing_date,
                fiscal_year=center.fiscal_year,
                accession_number=center.accession_number,
                rrf_score=hit.rrf_score,
                semantic_rank=hit.semantic_rank,
                lexical_rank=hit.lexical_rank,
                neighbors=neighbors,
            )
        )
    return passages


def fetch_chunk(connection: Connection, chunk_id: UUID) -> RetrievedPassage | None:
    from app.retrieval.retriever import RetrievedPassage

    row = connection.execute(FETCH_CHUNK, {"chunk_id": chunk_id}).one_or_none()
    if row is None:
        return None
    return _stored_passage(row, RetrievedPassage)


def fetch_surrounding(connection: Connection, chunk_id: UUID) -> list[RetrievedPassage]:
    from app.retrieval.retriever import RetrievedPassage

    rows = connection.execute(FETCH_SURROUNDING, {"chunk_id": chunk_id}).all()
    return [_stored_passage(row, RetrievedPassage) for row in rows]


def _stored_passage(
    row: object, passage_type: type[RetrievedPassage]
) -> RetrievedPassage:
    return passage_type(
        chunk_id=_uuid(row.id),
        document_id=_uuid(row.document_id),
        chunk_index=row.chunk_index,
        content=row.content,
        token_count=row.token_count,
        metadata=dict(row.metadata or {}),
        ticker=row.ticker,
        company_name=row.company_name,
        filing_type=row.filing_type,
        filing_date=row.filing_date,
        fiscal_year=row.fiscal_year,
        accession_number=row.accession_number,
        rrf_score=0.0,
        semantic_rank=None,
        lexical_rank=None,
        neighbors=(),
    )


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in values) + "]"


def _ids(result: Result) -> list[UUID]:
    return [_uuid(row.id) for row in result.all()]


def _uuid(value: UUID | str) -> UUID:
    if isinstance(value, UUID):
        return value
    return UUID(str(value))
