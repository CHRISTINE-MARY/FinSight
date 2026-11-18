"""One chat turn: retrieve, answer, then reject uncited claims."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.assistant.agent import get_agent
from app.assistant.deps import DocumentAgentDeps
from app.auth.dependencies import CurrentUser
from app.chat.messages import assistant_parts
from app.database.chats import CitationInsert, ThreadRecord
from app.grounding.validator import validate
from app.retrieval.retriever import DocumentRetriever, RetrievedPassage


class RetrievalError(Exception):
    pass


class AssistantError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class CompletedTurn:
    answer: str
    parts: list[dict[str, object]]
    citations: tuple[CitationInsert, ...]


async def run_turn(
    user: CurrentUser, thread: ThreadRecord, question: str
) -> CompletedTurn:
    retriever = DocumentRetriever()
    deps = DocumentAgentDeps(
        user_id=str(user.id),
        thread_id=str(thread.id),
        retriever=retriever,
    )
    try:
        passages = await asyncio.to_thread(retriever.search, question)
    except Exception as exc:
        raise RetrievalError("Retrieval failed") from exc
    for passage in passages:
        deps.retrieved[passage.chunk_id] = passage
    try:
        result = await get_agent().run(_prompt(question, passages), deps=deps)
    except Exception as exc:
        raise AssistantError("The assistant could not complete this answer") from exc
    validated = validate(result.output, deps.retrieved)
    citation_data = [citation.data for citation in validated.citations]
    return CompletedTurn(
        answer=validated.answer,
        parts=assistant_parts(validated.answer, citation_data),
        citations=tuple(
            CitationInsert(chunk_id=citation.chunk_id, excerpt=citation.excerpt)
            for citation in validated.citations
        ),
    )


def _prompt(question: str, passages: list[RetrievedPassage]) -> str:
    if passages:
        blocks = "\n\n---\n\n".join(_block(passage) for passage in passages)
    else:
        blocks = "No passages were retrieved."
    return f"Question:\n{question}\n\nRetrieved passages:\n{blocks}"


def _block(passage: RetrievedPassage) -> str:
    page = passage.metadata.get("page")
    section = passage.metadata.get("section")
    return (
        f"chunk_id: {passage.chunk_id}\n"
        f"ticker: {passage.ticker}\n"
        f"company: {passage.company_name}\n"
        f"filing: {passage.filing_type} FY{passage.fiscal_year} "
        f"filed {passage.filing_date.isoformat()}\n"
        f"page: {page}\n"
        f"section: {section}\n"
        f"content:\n{passage.content}"
    )
