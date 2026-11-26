"""Read source documents and write document_chunks."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import create_engine, delete, func, select, text, update
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.engine.url import make_url

from app.config import settings
from app.database.models.document_chunk import (
    EMBEDDING_STATUS_EMBEDDED,
    EMBEDDING_STATUS_FAILED,
    EMBEDDING_STATUS_PENDING,
    DocumentChunk,
)
from app.database.models.source_document import SourceDocument
from app.ingestion.metadata import FilingContext, ingestion_action


@dataclass(frozen=True, slots=True)
class ChunkWrite:
    chunk_index: int
    content: str
    token_count: int
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class ChunkToEmbed:
    id: UUID
    content: str


@dataclass(frozen=True, slots=True)
class StoreResult:
    action: str
    document_id: UUID
    chunk_count: int


_engine: Engine | None = None


def database_engine() -> Engine:
    global _engine
    if _engine is None:
        parsed = make_url(settings.database_url)
        if parsed.drivername != "postgresql+psycopg":
            parsed = parsed.set(drivername="postgresql+psycopg")
        _engine = create_engine(
            parsed.render_as_string(hide_password=False), pool_pre_ping=True
        )
    return _engine


def fetch_filing(accession_number: str) -> FilingContext:
    engine = database_engine()
    with engine.connect() as connection:
        return _filing(connection, accession_number)


def store_chunks(accession_number: str, rows: list[ChunkWrite]) -> StoreResult:
    """Insert chunks, or skip when this accession was already stored in full."""
    engine = database_engine()
    with engine.begin() as connection:
        filing = _filing(connection, accession_number)
        stored = _chunk_count(connection, filing.id)
        action = ingestion_action(stored, len(rows))
        if action == "skip":
            return StoreResult(
                action="skipped", document_id=filing.id, chunk_count=stored
            )
        connection.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == filing.id)
        )
        if rows:
            connection.execute(
                DocumentChunk.__table__.insert(),
                [
                    {
                        "document_id": filing.id,
                        "chunk_index": row.chunk_index,
                        "content": row.content,
                        "token_count": row.token_count,
                        "embedding": None,
                        "embedding_status": EMBEDDING_STATUS_PENDING,
                        "metadata": row.metadata,
                    }
                    for row in rows
                ],
            )
        return StoreResult(
            action="stored", document_id=filing.id, chunk_count=len(rows)
        )


def chunk_status_counts() -> dict[str, int]:
    """Count stored chunks by embedding_status."""
    engine = database_engine()
    with engine.connect() as connection:
        rows = connection.execute(
            select(DocumentChunk.embedding_status, func.count()).group_by(
                DocumentChunk.embedding_status
            )
        ).all()
    return {status: int(count) for status, count in rows}


def fetch_chunks_for_embedding(limit: int) -> list[ChunkToEmbed]:
    """Return chunks that still need a vector, oldest filing and index first."""
    engine = database_engine()
    with engine.connect() as connection:
        rows = connection.execute(
            select(DocumentChunk.id, DocumentChunk.content)
            .where(DocumentChunk.embedding_status == EMBEDDING_STATUS_PENDING)
            .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
            .limit(limit)
        ).all()
    return [ChunkToEmbed(id=row.id, content=row.content) for row in rows]


def save_embeddings(rows: list[tuple[UUID, list[float]]]) -> None:
    """Write vectors and mark those chunks embedded."""
    engine = database_engine()
    with engine.begin() as connection:
        for chunk_id, embedding in rows:
            connection.execute(
                update(DocumentChunk)
                .where(DocumentChunk.id == chunk_id)
                .values(
                    embedding=embedding,
                    embedding_status=EMBEDDING_STATUS_EMBEDDED,
                )
            )


def mark_embedding_failed(chunk_ids: list[UUID]) -> None:
    if not chunk_ids:
        return
    engine = database_engine()
    with engine.begin() as connection:
        connection.execute(
            update(DocumentChunk)
            .where(DocumentChunk.id.in_(chunk_ids))
            .values(embedding_status=EMBEDDING_STATUS_FAILED)
        )


def chunk_evidence(
    document_id: UUID, chunk_index: int, query: str
) -> dict[str, object]:
    """Read back one stored chunk, including the generated search vector."""
    statement = text(
        """
        SELECT
            content,
            token_count,
            metadata,
            vector_dims(embedding) AS dimensions,
            search_vector IS NOT NULL AS has_search_vector,
            (search_vector @@ plainto_tsquery('english', :query)) AS matches_query
        FROM document_chunks
        WHERE document_id = :document_id AND chunk_index = :chunk_index
        """
    )
    engine = database_engine()
    with engine.connect() as connection:
        row = connection.execute(
            statement,
            {"document_id": document_id, "chunk_index": chunk_index, "query": query},
        ).one()
    return {
        "content": row.content,
        "token_count": row.token_count,
        "metadata": row.metadata,
        "dimensions": row.dimensions,
        "has_search_vector": row.has_search_vector,
        "matches_query": row.matches_query,
    }


def _filing(connection: Connection, accession_number: str) -> FilingContext:
    row = connection.execute(
        select(
            SourceDocument.id,
            SourceDocument.ticker,
            SourceDocument.company_name,
            SourceDocument.filing_type,
            SourceDocument.filing_date,
            SourceDocument.fiscal_year,
            SourceDocument.accession_number,
        ).where(SourceDocument.accession_number == accession_number)
    ).one_or_none()
    if row is None:
        raise LookupError(f"No source document for accession {accession_number}")
    return FilingContext(
        id=row.id,
        ticker=row.ticker,
        company_name=row.company_name,
        filing_type=row.filing_type,
        filing_date=row.filing_date,
        fiscal_year=row.fiscal_year,
        accession_number=row.accession_number,
    )


def _chunk_count(connection: Connection, document_id: UUID) -> int:
    count = connection.execute(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
    ).scalar_one()
    return int(count)
