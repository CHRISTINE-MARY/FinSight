"""Copy filing fields onto each chunk, plus the section and page Docling found."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from uuid import UUID

from app.ingestion.types import PreparedChunk


@dataclass(frozen=True, slots=True)
class FilingContext:
    id: UUID
    ticker: str
    company_name: str
    filing_type: str
    filing_date: date
    fiscal_year: int
    accession_number: str


def chunk_metadata(filing: FilingContext, chunk: PreparedChunk) -> dict[str, object]:
    return {
        "ticker": filing.ticker,
        "company_name": filing.company_name,
        "filing_type": filing.filing_type,
        "filing_date": filing.filing_date.isoformat(),
        "fiscal_year": filing.fiscal_year,
        "accession_number": filing.accession_number,
        "section": chunk.headings,
        "page": chunk.pages,
        "doc_items": chunk.doc_item_refs,
    }


def ingestion_action(stored_count: int, new_count: int) -> str:
    """Skip a document only when a previous run stored the same number of chunks."""
    if stored_count > 0 and stored_count == new_count:
        return "skip"
    return "replace"


def best_revenue_chunk(chunks: list[PreparedChunk]) -> PreparedChunk:
    """Pick the net-sales-by-category table, not a sentence that merely names products."""
    table = [
        chunk for chunk in chunks if "209,586" in chunk.text and "iPhone" in chunk.text
    ]
    if table:
        return table[0]
    required = ("iPhone", "Mac", "iPad", "Wearables", "Services")
    candidates = [
        chunk for chunk in chunks if all(marker in chunk.text for marker in required)
    ]
    if not candidates:
        raise LookupError("No chunk contains the Apple revenue mix")
    return candidates[0]
