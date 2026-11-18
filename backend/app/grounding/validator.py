"""Reject answers whose citations are not from this request's passages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from uuid import UUID

from app.assistant.outputs import GroundedAnswer
from app.retrieval.retriever import RetrievedPassage

NOT_ENOUGH_EVIDENCE = "not enough evidence"
CORPUS_PHRASE = "does not contain enough evidence"
_EXCERPT_LIMIT = 500


class GroundingError(Exception):
    """The answer cited something this request did not retrieve."""


@dataclass(frozen=True, slots=True)
class ValidatedCitation:
    chunk_id: UUID
    excerpt: str
    data: dict[str, object]


@dataclass(frozen=True, slots=True)
class ValidatedAnswer:
    answer: str
    citations: tuple[ValidatedCitation, ...]


def says_not_enough(answer: str) -> bool:
    lowered = " ".join(answer.lower().split())
    return NOT_ENOUGH_EVIDENCE in lowered or CORPUS_PHRASE in lowered


def validate(
    answer: GroundedAnswer, retrieved: Mapping[UUID, RetrievedPassage]
) -> ValidatedAnswer:
    """Keep the answer only when every citation is a retrieved passage."""
    text = answer.answer.strip()
    if not text:
        raise GroundingError("The answer is empty")
    if not answer.citations and not says_not_enough(text):
        raise GroundingError("The answer has no citation")

    citations: list[ValidatedCitation] = []
    seen: set[tuple[UUID, str]] = set()
    for citation in answer.citations:
        passage = retrieved.get(citation.chunk_id)
        if passage is None:
            raise GroundingError(f"Chunk {citation.chunk_id} was not retrieved")
        excerpt = _excerpt(passage.content, citation.excerpt)
        key = (citation.chunk_id, excerpt)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            ValidatedCitation(
                chunk_id=citation.chunk_id,
                excerpt=excerpt,
                data=_citation_data(passage, excerpt),
            )
        )
    return ValidatedAnswer(answer=text, citations=tuple(citations))


def _excerpt(content: str, excerpt: str) -> str:
    cleaned = excerpt.strip()
    if not cleaned:
        raise GroundingError("A citation is missing its excerpt")
    if cleaned in content:
        stored = cleaned
    elif _squash(cleaned) in _squash(content):
        stored = _squash(cleaned)
    else:
        raise GroundingError("Excerpt is not in the retrieved passage")
    if len(stored) > _EXCERPT_LIMIT:
        stored = stored[:_EXCERPT_LIMIT].rstrip()
    return stored


def _squash(value: str) -> str:
    return " ".join(value.split())


def _citation_data(passage: RetrievedPassage, excerpt: str) -> dict[str, object]:
    return {
        "chunkId": str(passage.chunk_id),
        "ticker": passage.ticker,
        "companyName": passage.company_name,
        "filingType": passage.filing_type,
        "filingDate": passage.filing_date.isoformat(),
        "fiscalYear": passage.fiscal_year,
        "accessionNumber": passage.accession_number,
        "page": _text(passage.metadata.get("page")),
        "section": _text(passage.metadata.get("section")),
        "excerpt": excerpt,
    }


def _text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        joined = ", ".join(parts)
        return joined or None
    text = str(value).strip()
    return text or None
