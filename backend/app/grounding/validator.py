"""Reject answers whose citations are not from this request's passages."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from uuid import UUID

from app.assistant.outputs import GroundedAnswer
from app.grounding.tables import (
    YEAR_HEADER_SPAN,
    GroundingError,
    locate_line,
    locator_in_text,
    reject_wrong_column,
)
from app.retrieval.retriever import RetrievedPassage

NOT_ENOUGH_EVIDENCE = "not enough evidence"
CORPUS_PHRASE = "does not contain enough evidence"
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
_WORD = re.compile(r"[A-Za-z]{4,}")


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
    answer: GroundedAnswer,
    retrieved: Mapping[UUID, RetrievedPassage],
    question: str = "",
) -> ValidatedAnswer:
    """Keep the answer only when every citation is a retrieved passage.

    The model's excerpt is ignored. The stored excerpt is an exact slice of
    the named chunk. A year in the question has to match that column.
    """
    text = answer.answer.strip()
    if not text:
        raise GroundingError("The answer is empty")
    if not answer.citations and not says_not_enough(text):
        raise GroundingError("The answer has no citation")

    citations: list[ValidatedCitation] = []
    cited: list[tuple[str, list[str]]] = []
    seen: set[tuple[UUID, str]] = set()
    for citation in answer.citations:
        passage = retrieved.get(citation.chunk_id)
        if passage is None:
            raise GroundingError(f"Chunk {citation.chunk_id} was not retrieved")
        try:
            excerpt = locate_line(
                passage.content, citation.label, citation.figure, text
            )
        except GroundingError as exc:
            raise _classify_missing(
                exc, retrieved, passage.chunk_id, citation.label, citation.figure
            ) from exc
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
        cited.append((excerpt, _context_lines(passage, retrieved), passage.content))
    if citations and not says_not_enough(text):
        if not _supported(text, citations):
            raise GroundingError("The cited passage does not support the answer")
        try:
            reject_wrong_column(question, text, cited)
        except GroundingError as exc:
            raise _classify_missing(exc, retrieved, None, "", "") from exc
    return ValidatedAnswer(answer=text, citations=tuple(citations))


def _classify_missing(
    exc: GroundingError,
    retrieved: Mapping[UUID, RetrievedPassage],
    cited_id: UUID | None,
    label: str,
    figure: str,
) -> GroundingError:
    """A fact in some other retrieved chunk was cited wrong. One in none was invented."""
    message = str(exc)
    if not _is_missing_fact(message):
        return exc
    others = [
        passage.content
        for chunk_id, passage in retrieved.items()
        if chunk_id != cited_id
    ]
    if _fact_elsewhere(message, label, figure, others):
        return GroundingError(f"Citation mistake: {message}")
    return GroundingError(f"Hallucination: {message}")


def _is_missing_fact(message: str) -> bool:
    return (
        message == "Locator is not in the retrieved passage"
        or message.endswith(" is not in the cited passage")
    )


def _fact_elsewhere(message: str, label: str, figure: str, others: list[str]) -> bool:
    if message == "Locator is not in the retrieved passage":
        return any(locator_in_text(content, label, figure) for content in others)
    token = message[: -len(" is not in the cited passage")]
    folded = token.casefold()
    return any(folded in content.casefold() for content in others)


def _supported(answer: str, citations: list[ValidatedCitation]) -> bool:
    """A chunk id is not evidence unless the answer uses text from that chunk."""
    haystack = "\n".join(citation.excerpt for citation in citations).casefold()
    for number in _NUMBER.findall(answer):
        if number.casefold() in haystack:
            return True
    for word in _WORD.findall(answer):
        if word.casefold() in haystack:
            return True
    return False


def _context_lines(
    passage: RetrievedPassage, retrieved: Mapping[UUID, RetrievedPassage]
) -> list[str]:
    """Table lines from this chunk and nearby chunks of the same filing."""
    blocks: list[tuple[int, str]] = []
    seen: set[UUID] = set()

    def add(index: int, chunk_id: UUID, content: str) -> None:
        if chunk_id in seen:
            return
        seen.add(chunk_id)
        blocks.append((index, content))

    add(passage.chunk_index, passage.chunk_id, passage.content)
    for neighbor in passage.neighbors:
        if abs(neighbor.chunk_index - passage.chunk_index) <= 1:
            add(neighbor.chunk_index, neighbor.chunk_id, neighbor.content)
    for other in retrieved.values():
        if other.document_id != passage.document_id:
            continue
        if abs(other.chunk_index - passage.chunk_index) > YEAR_HEADER_SPAN:
            continue
        add(other.chunk_index, other.chunk_id, other.content)
    lines: list[str] = []
    for _, content in sorted(blocks):
        for line in content.splitlines():
            if line.strip().startswith("|") and "---" not in line:
                lines.append(line)
    return lines


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
        "content": passage.content,
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
