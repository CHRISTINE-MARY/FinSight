"""Typed answer the filing assistant must return."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: UUID = Field(description="Id of a chunk returned for this request")
    excerpt: str = Field(description="Contiguous text copied from that chunk")


class SourcePassage(BaseModel):
    """Filing passage the server attaches after citation checks."""

    chunk_id: UUID
    ticker: str
    company_name: str
    filing_type: str
    filing_date: str
    fiscal_year: int
    accession_number: str
    page: str | None = None
    section: str | None = None
    content: str


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    cited_passages: list[SourcePassage] = Field(default_factory=list)
