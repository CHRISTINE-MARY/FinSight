from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Date, DateTime, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.models.base import Base

if TYPE_CHECKING:
    from app.database.models.document_chunk import DocumentChunk


class SourceDocument(Base):
    __tablename__ = "source_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, server_default=func.gen_random_uuid()
    )
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    company_name: Mapped[str] = mapped_column(Text)
    cik: Mapped[str] = mapped_column(String(10))
    filing_type: Mapped[str] = mapped_column(String(16))
    filing_date: Mapped[date] = mapped_column(Date)
    report_date: Mapped[date | None] = mapped_column(Date)
    fiscal_year: Mapped[int] = mapped_column(Integer)
    accession_number: Mapped[str] = mapped_column(String(32), unique=True)
    source_url: Mapped[str] = mapped_column(Text)
    markdown: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
