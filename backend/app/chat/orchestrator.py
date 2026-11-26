"""One chat turn: retrieve, answer, then reject uncited claims."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from app.assistant.agent import get_agent
from app.assistant.deps import DocumentAgentDeps, StatusReport
from app.assistant.outputs import compose_answer
from app.auth.dependencies import CurrentUser
from app.chat.messages import assistant_parts
from app.chat.status import threadsafe_report
from app.database.chats import (
    CitationInsert,
    StoredMessage,
    ThreadRecord,
    list_messages,
)
from app.grounding.validator import validate
from app.retrieval.retriever import DocumentRetriever, RetrievedPassage


class RetrievalError(Exception):
    pass


class AssistantError(Exception):
    pass


_PRIOR_QUESTIONS = 6
_COMPANIES = (
    (("apple", "apples", "aapl"), "Apple"),
    (("amazon", "amzn"), "Amazon"),
    (("microsoft", "msft"), "Microsoft"),
    (("nvidia", "nvda"), "NVIDIA"),
    (("alphabet", "google", "googl", "goog"), "Alphabet"),
)
_THREAD_MARKERS = (
    "prev question",
    "previous question",
    "talking about",
    "which company",
    "what company",
    "which one",
)
_THREAD_FILLER = {
    "a",
    "about",
    "an",
    "and",
    "company",
    "i",
    "me",
    "or",
    "the",
    "this",
    "wait",
    "was",
    "which",
}


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
        prior = _recent_user_questions(await list_messages(user, thread.id))
    except Exception as exc:
        raise RetrievalError("Retrieval failed") from exc
    if _is_thread_question(question):
        await note("retrieve", "Reading the earlier questions in this thread")
        answer = _thread_answer(question, prior)
        return CompletedTurn(
            answer=answer,
            parts=assistant_parts(answer, [], notes),
            citations=(),
        )
    try:
        passages = await asyncio.to_thread(
            retriever.search,
            question,
            prior_questions=prior,
            progress=threadsafe_report(note),
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
        output = await _complete(_prompt(question, passages, prior), deps, note)
    except Exception as exc:
        raise AssistantError("The assistant could not complete this answer") from exc
    output = output.model_copy(
        update={"answer": compose_answer(output.answer, output.table)}
    )
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


def _prompt(
    question: str, passages: list[RetrievedPassage], prior_questions: list[str]
) -> str:
    if passages:
        blocks = "\n\n---\n\n".join(_block(passage) for passage in passages)
    else:
        blocks = "No passages were retrieved."
    table_line = ""
    if _wants_table(question):
        table_line = (
            "\n\nFill the table field. One row per segment or item, one column "
            "per fiscal year in the question. The answer field is only the lead sentence."
        )
    history = _history_block(prior_questions)
    return f"Question:\n{question}{table_line}{history}\n\nRetrieved passages:\n{blocks}"


def _history_block(prior_questions: list[str]) -> str:
    if not prior_questions:
        return ""
    lines = "\n".join(f"- {item}" for item in prior_questions)
    return (
        "\n\nThe analyst already asked:\n"
        f"{lines}\n"
        "If this question omits the company or the metric, keep them from those "
        "earlier questions unless it names a different company."
    )


def _is_thread_question(question: str) -> bool:
    lowered = question.lower()
    if any(marker in lowered for marker in _THREAD_MARKERS):
        return True
    tokens = re.findall(r"[a-z0-9]+", lowered)
    if len(_companies_in(question)) < 2:
        return False
    return all(token in _THREAD_FILLER or _company_named(token) for token in tokens)


def _thread_answer(question: str, prior_questions: list[str]) -> str:
    if not prior_questions:
        return "This thread has no earlier question."
    mentioned: list[str] = []
    for text in prior_questions:
        for name in _companies_in(text):
            if name not in mentioned:
                mentioned.append(name)
    if not mentioned:
        return "Your earlier questions did not name a company."
    asked = _companies_in(question)
    if len(asked) >= 2:
        overlap = [name for name in asked if name in mentioned]
        if not overlap:
            return f"Your earlier questions did not name {_join(asked)}."
        return f"Your earlier questions were about {_join(overlap)}."
    return f"Your earlier questions were about {mentioned[-1]}."


def _companies_in(text: str) -> list[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return [name for aliases, name in _COMPANIES if tokens & set(aliases)]


def _company_named(token: str) -> bool:
    return any(token in aliases for aliases, _name in _COMPANIES)


def _join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _recent_user_questions(messages: list[StoredMessage]) -> list[str]:
    questions: list[str] = []
    for message in messages:
        if message.role != "user":
            continue
        text = _part_text(message.parts)
        if text:
            questions.append(text)
    return questions[-_PRIOR_QUESTIONS:]


def _part_text(parts: list[dict[str, object]]) -> str:
    chunks: list[str] = []
    for part in parts:
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    return "".join(chunks).strip()


def _wants_table(question: str) -> bool:
    lowered = question.lower()
    if any(word in lowered for word in ("table", "compare", "comparison")):
        return True
    return "mix" in lowered and "year" in lowered


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
