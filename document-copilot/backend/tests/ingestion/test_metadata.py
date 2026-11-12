from datetime import date
from uuid import UUID

import pytest

from app.ingestion.metadata import (
    FilingContext,
    best_revenue_chunk,
    chunk_metadata,
    ingestion_action,
)
from app.ingestion.types import PreparedChunk

FILING = FilingContext(
    id=UUID("11111111-1111-4111-8111-111111111111"),
    ticker="AAPL",
    company_name="Apple Inc.",
    filing_type="10-K",
    filing_date=date(2025, 10, 31),
    fiscal_year=2025,
    accession_number="0000320193-25-000079",
)


def _chunk(
    *,
    index: int = 0,
    text: str = "Net sales\niPhone Mac iPad Services",
    headings: list[str] | None = None,
) -> PreparedChunk:
    return PreparedChunk(
        index=index,
        text=text,
        token_count=12,
        headings=["Net sales"] if headings is None else headings,
        pages=[42],
        doc_item_refs=["#/tables/3"],
    )


def test_metadata_copies_filing_fields_and_docling_location() -> None:
    metadata = chunk_metadata(FILING, _chunk())

    assert metadata["ticker"] == "AAPL"
    assert metadata["company_name"] == "Apple Inc."
    assert metadata["filing_type"] == "10-K"
    assert metadata["filing_date"] == "2025-10-31"
    assert metadata["fiscal_year"] == 2025
    assert metadata["accession_number"] == "0000320193-25-000079"
    assert metadata["section"] == ["Net sales"]
    assert metadata["page"] == [42]
    assert metadata["doc_items"] == ["#/tables/3"]


def test_ingestion_skips_only_a_complete_existing_set() -> None:
    assert ingestion_action(stored_count=0, new_count=10) == "replace"
    assert ingestion_action(stored_count=1, new_count=10) == "replace"
    assert ingestion_action(stored_count=10, new_count=10) == "skip"


def test_revenue_chunk_prefers_the_passage_with_the_product_mix() -> None:
    partial = _chunk(index=1, text="Greater China net sales of iPhone, Services, iPad, Mac and Wearables")
    mix = _chunk(index=2, text="iPhone $ 209,586 Mac iPad Wearables Services Total net sales")
    chosen = best_revenue_chunk([partial, mix])

    assert chosen.index == 2


def test_revenue_chunk_requires_iphone_and_services() -> None:
    with pytest.raises(LookupError, match="revenue mix"):
        best_revenue_chunk([_chunk(text="Mac and iPad only")])
