from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.models.base import Base

if TYPE_CHECKING:
    from app.database.models.message_citation import MessageCitation
    from app.database.models.source_document import SourceDocument

# Stored width is fixed so migrations do not follow model drift.
# pgvector HNSW indexes stop at 2000 dimensions. Nemotron 3.5 Embed 1B
# returns 2048 values; ingestion keeps the first 1536 and L2-normalizes them.
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_STATUS_PENDING = "pending"
EMBEDDING_STATUS_EMBEDDED = "embedded"
EMBEDDING_STATUS_FAILED = "failed"


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_id_chunk_index",
        ),
        CheckConstraint(
            "embedding_status IN ('pending', 'embedded', 'failed')",
            name="ck_document_chunks_embedding_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, server_default=func.gen_random_uuid()
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("source_documents.id", ondelete="CASCADE"),
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=True
    )
    embedding_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'pending'")
    )
    # Postgres fills this from content so lexical search cannot drift from the passage.
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
    )
    # Filing context copied onto the passage: ticker, company_name, filing_type,
    # filing_date, fiscal_year, accession_number, page, section, source offsets.
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    document: Mapped[SourceDocument] = relationship(back_populates="chunks")
    citations: Mapped[list[MessageCitation]] = relationship(back_populates="chunk")
