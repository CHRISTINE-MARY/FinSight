from datetime import date
from uuid import UUID

import pytest

from app.assistant.outputs import Citation, GroundedAnswer
from app.grounding.validator import GroundingError, validate
from app.retrieval.retriever import RetrievedPassage

CHUNK_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
OTHER_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
CONTENT = "iPhone net sales were $201,183 million."


def test_citation_must_come_from_a_retrieved_passage() -> None:
    answer = GroundedAnswer(
        answer="iPhone sales were $201,183 million.",
        citations=[Citation(chunk_id=OTHER_ID, excerpt="iPhone net sales")],
    )

    with pytest.raises(GroundingError, match="not retrieved"):
        validate(answer, {CHUNK_ID: _passage()})


def test_factual_answer_without_a_citation_is_rejected() -> None:
    answer = GroundedAnswer(answer="iPhone sales grew.")

    with pytest.raises(GroundingError, match="no citation"):
        validate(answer, {CHUNK_ID: _passage()})


def test_not_enough_evidence_may_have_no_citations() -> None:
    answer = GroundedAnswer(answer="The corpus does not contain enough evidence.")

    validated = validate(answer, {})

    assert validated.citations == ()


def test_valid_citation_uses_the_retrieved_filing_not_the_model() -> None:
    answer = GroundedAnswer(
        answer="iPhone net sales were $201,183 million.",
        citations=[
            Citation(
                chunk_id=CHUNK_ID, excerpt="iPhone net sales were $201,183 million."
            )
        ],
    )

    validated = validate(answer, {CHUNK_ID: _passage()})

    assert validated.citations[0].chunk_id == CHUNK_ID
    assert validated.citations[0].data["ticker"] == "AAPL"
    assert validated.citations[0].data["filingType"] == "10-K"
    assert validated.citations[0].data["page"] == "42"
    assert validated.citations[0].data["companyName"] == "Apple Inc."


def test_excerpt_missing_from_the_passage_is_rejected() -> None:
    answer = GroundedAnswer(
        answer="Services grew.",
        citations=[Citation(chunk_id=CHUNK_ID, excerpt="Services grew 14 percent")],
    )

    with pytest.raises(GroundingError, match="Excerpt"):
        validate(answer, {CHUNK_ID: _passage()})


def _passage() -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=CHUNK_ID,
        document_id=UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc"),
        chunk_index=4,
        content=CONTENT,
        token_count=12,
        metadata={"page": 42, "section": ["Net sales"]},
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type="10-K",
        filing_date=date(2024, 11, 1),
        fiscal_year=2024,
        accession_number="0000320193-24-000123",
        rrf_score=0.03,
        semantic_rank=1,
        lexical_rank=1,
        neighbors=(),
    )
