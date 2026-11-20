"""One chat turn: retrieve, answer, then reject uncited claims."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.assistant.agent import get_agent
from app.assistant.deps import DocumentAgentDeps, StatusReport
from app.auth.dependencies import CurrentUser
from app.chat.messages import assistant_parts
from app.chat.status import threadsafe_report
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
    user: CurrentUser,
    thread: ThreadRecord,
    question: str,
    *,
    report: StatusReport | None = None,
) -> CompletedTurn:
    notes: list[dict[str, str]] = []

    async def note(phase: str, detail: str) -> None:
        notes.append({"phase": phase, "detail": detail})
        if report is not None:
            await report(phase, detail)

    retriever = DocumentRetriever()
    deps = DocumentAgentDeps(
        user_id=str(user.id),
        thread_id=str(thread.id),
        retriever=retriever,
        report=note,
    )
    try:
        passages = await asyncio.to_thread(
            retriever.search, question, progress=threadsafe_report(note)
        )
    except Exception as exc:
        raise RetrievalError("Retrieval failed") from exc
    for passage in passages:
        deps.retrieved[passage.chunk_id] = passage
    if passages:
        labels = ", ".join(_passage_label(passage) for passage in passages)
        noun = "passage" if len(passages) == 1 else "passages"
        await note("retrieve", f"Retrieved {len(passages)} {noun}: {labels}")
    try:
        output = await _complete(_prompt(question, passages), deps, note)
    except Exception as exc:
        raise AssistantError("The assistant could not complete this answer") from exc
    await note("validate", "Checking that every citation maps to a retrieved passage")
    validated = validate(output, deps.retrieved)
    count = len(validated.citations)
    noun = "citation" if count == 1 else "citations"
    await note("validate", f"Grounded with {count} {noun}")
    citation_data = [citation.data for citation in validated.citations]
    return CompletedTurn(
        answer=validated.answer,
        parts=assistant_parts(validated.answer, citation_data, notes),
        citations=tuple(
            CitationInsert(chunk_id=citation.chunk_id, excerpt=citation.excerpt)
            for citation in validated.citations
        ),
    )


async def _complete(prompt: str, deps: DocumentAgentDeps, note: StatusReport):
    round_index = 0
    async with get_agent().iter(prompt, deps=deps) as agent_run:
        async for node in agent_run:
            name = type(node).__name__
            if name == "ModelRequestNode":
                round_index += 1
                await note("model", f"Model round {round_index}: waiting for the model")
            elif name == "CallToolsNode":
                labels = _tool_labels(node)
                if labels:
                    joined = ", ".join(labels)
                    await note("model", f"Model round {round_index}: calling {joined}")
                else:
                    await note(
                        "model", f"Model round {round_index}: model returned text"
                    )
    if agent_run.result is None:
        raise AssistantError("The assistant returned no answer")
    return agent_run.result.output


def _tool_labels(node: object) -> list[str]:
    response = getattr(node, "model_response", None)
    parts = getattr(response, "parts", ()) if response is not None else ()
    labels: list[str] = []
    for part in parts:
        if getattr(part, "part_kind", None) != "tool-call":
            continue
        labels.append(str(getattr(part, "tool_name", "tool")))
    return labels


def _prompt(question: str, passages: list[RetrievedPassage]) -> str:
    if passages:
        blocks = "\n\n---\n\n".join(_block(passage) for passage in passages)
    else:
        blocks = "No passages were retrieved."
    return f"Question:\n{question}\n\nRetrieved passages:\n{blocks}"


def _passage_label(passage: RetrievedPassage) -> str:
    return f"{passage.ticker} {passage.filing_type} FY{passage.fiscal_year}"


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
