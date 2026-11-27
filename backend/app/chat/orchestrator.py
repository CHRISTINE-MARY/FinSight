"""One chat turn: retrieve, answer, then reject uncited claims."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from uuid import UUID

from app.assistant.agent import get_agent
from app.assistant.deps import DocumentAgentDeps, StatusReport
from app.assistant.outputs import Citation, GroundedAnswer, ModelAnswer, compose_answer
from app.auth.dependencies import CurrentUser
from app.chat.messages import assistant_parts
from app.chat.refusal import Filer, evidence_sufficient, out_of_scope
from app.chat.status import threadsafe_report
from app.database.chats import (
    CitationInsert,
    StoredMessage,
    ThreadRecord,
    list_messages,
)
from app.database.documents import list_filers
from app.grounding.tables import YEAR_HEADER_SPAN, GroundingError, year_column_lines
from app.grounding.validator import validate
from app.ingestion.metadata import section_name
from app.retrieval.retriever import DocumentRetriever, NeighborChunk, RetrievedPassage
from app.retrieval.scope import filing_header


class RetrievalError(Exception):
    pass


class AssistantError(Exception):
    pass


_PRIOR_QUESTIONS = 6
_REFUSAL = "The corpus does not contain enough evidence."
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
    _remember(deps.retrieved, passages)
    try:
        filers = [Filer(ticker, name) for ticker, name in list_filers()]
    except Exception as exc:
        raise RetrievalError("Retrieval failed") from exc
    if out_of_scope(question, filers):
        await note("validate", "That company is not in the corpus")
        return _refused(notes)
    if not evidence_sufficient(question, passages, filers):
        await note("validate", "The retrieved passages do not cover the question")
        return _refused(notes)
    if passages:
        labels = ", ".join(_passage_label(passage) for passage in passages)
        noun = "passage" if len(passages) == 1 else "passages"
        await note("retrieve", f"Retrieved {len(passages)} {noun}: {labels}")
    _number_passages(deps, passages)
    prompt = _prompt(
        question,
        passages,
        prior,
        deps.numbers,
        getattr(retriever, "unavailable", ()),
    )
    try:
        output = await _complete(prompt, deps, note)
    except Exception as exc:
        raise AssistantError("The assistant could not complete this answer") from exc
    await note("validate", "Checking that every citation maps to a retrieved passage")
    try:
        validated = _check(output, deps, question)
    except GroundingError as exc:
        await note("validate", f"Retrying: {exc}")
        try:
            output = await _complete(
                _retry_prompt(prompt, _claim_text(output), str(exc)), deps, note
            )
        except Exception as exc:
            raise AssistantError("The assistant could not complete this answer") from exc
        await note(
            "validate", "Checking that every citation maps to a retrieved passage"
        )
        validated = _check(output, deps, question)
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


def _check(
    output: GroundedAnswer | ModelAnswer,
    deps: DocumentAgentDeps,
    question: str,
):
    grounded = _as_grounded(output, deps.numbers)
    grounded = grounded.model_copy(
        update={"answer": compose_answer(grounded.answer, grounded.table)}
    )
    try:
        return validate(grounded, deps.retrieved, question)
    except GroundingError as exc:
        claim = _claim_text(output).strip()
        if claim and not str(exc).startswith("Claim "):
            raise GroundingError(f"Claim {claim!r} failed: {exc}") from exc
        raise


def _as_grounded(
    output: GroundedAnswer | ModelAnswer, numbers: dict[int, UUID]
) -> GroundedAnswer:
    if isinstance(output, GroundedAnswer):
        return output
    if not output.claims:
        return GroundedAnswer(answer=output.answer.strip(), table=output.table)
    citations: list[Citation] = []
    sentences: list[str] = []
    for claim in output.claims:
        text = claim.text.strip()
        chunk_id = numbers.get(claim.chunk)
        if not text:
            raise GroundingError(f"Claim for passage {claim.chunk} has no text")
        if chunk_id is None:
            raise GroundingError(
                f"Claim {text!r} cites passage {claim.chunk}, which was not retrieved"
            )
        sentences.append(text)
        citations.append(
            Citation(chunk_id=chunk_id, label=claim.label, figure=claim.figure)
        )
    return GroundedAnswer(
        answer=" ".join(sentences), table=output.table, citations=citations
    )


def _claim_text(output: GroundedAnswer | ModelAnswer) -> str:
    if isinstance(output, ModelAnswer):
        sentences = [claim.text.strip() for claim in output.claims if claim.text.strip()]
        if sentences:
            return " ".join(sentences)
        return output.answer
    return output.answer


def _retry_prompt(prompt: str, claim: str, reason: str) -> str:
    shown = claim.strip() or "(empty)"
    return (
        f"{prompt}\n\n"
        "The previous answer was rejected.\n"
        f"Claim: {shown}\n"
        f"Reason: {reason}\n"
        "Fix that claim. Every factual sentence must be its own claim, and "
        "each claim's chunk must be a passage number from this request."
    )


def _number_passages(
    deps: DocumentAgentDeps, passages: list[RetrievedPassage]
) -> None:
    """Number the retrieved chunks from 1, then any neighbor shown with them."""
    for passage in passages:
        deps.assign(passage.chunk_id)
    for passage in passages:
        for neighbor in passage.neighbors:
            deps.assign(neighbor.chunk_id)


def _prompt(
    question: str,
    passages: list[RetrievedPassage],
    prior_questions: list[str],
    numbers: dict[int, UUID],
    unavailable: object = (),
) -> str:
    if passages:
        blocks = "\n\n---\n\n".join(_prompt_blocks(passages, numbers))
    else:
        blocks = "No passages were retrieved."
    table_line = ""
    if _wants_table(question):
        table_line = (
            "\n\nFill the table field. One row per segment or item, one column "
            "per fiscal year in the question. The answer field is only the lead sentence."
        )
    history = _history_block(prior_questions)
    missing = _unavailable_block(unavailable)
    return (
        f"Question:\n{question}{table_line}{history}{missing}\n\n"
        f"Retrieved passages:\n{blocks}"
    )


def _unavailable_block(unavailable: object) -> str:
    if not isinstance(unavailable, (list, tuple)):
        return ""
    lines = [str(item).strip() for item in unavailable if str(item).strip()]
    if not lines:
        return ""
    body = "\n".join(f"- {line} Do not invent a figure for it." for line in lines)
    return f"\n\nUnavailable filings:\n{body}"


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


def _remember(
    retrieved: dict[UUID, RetrievedPassage], passages: list[RetrievedPassage]
) -> None:
    for passage in passages:
        retrieved[passage.chunk_id] = passage
    for passage in passages:
        for neighbor in passage.neighbors:
            retrieved.setdefault(neighbor.chunk_id, _neighbor_passage(passage, neighbor))


def _refused(notes: list[dict[str, str]]) -> CompletedTurn:
    return CompletedTurn(
        answer=_REFUSAL,
        parts=assistant_parts(_REFUSAL, [], notes),
        citations=(),
    )


def _prompt_blocks(
    passages: list[RetrievedPassage], numbers: dict[int, UUID]
) -> list[str]:
    by_chunk = {chunk_id: number for number, chunk_id in numbers.items()}
    ranked = {passage.chunk_id for passage in passages}
    blocks: list[str] = []
    for passage in passages:
        blocks.append(_block(passage, passages, by_chunk[passage.chunk_id]))
        for neighbor in passage.neighbors:
            if neighbor.chunk_id in ranked:
                continue
            blocks.append(
                _neighbor_block(
                    passage, neighbor, passages, by_chunk[neighbor.chunk_id]
                )
            )
    return blocks


def _neighbor_passage(
    passage: RetrievedPassage, neighbor: NeighborChunk
) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=neighbor.chunk_id,
        document_id=passage.document_id,
        chunk_index=neighbor.chunk_index,
        content=neighbor.content,
        token_count=0,
        metadata={},
        ticker=passage.ticker,
        company_name=passage.company_name,
        filing_type=passage.filing_type,
        filing_date=passage.filing_date,
        fiscal_year=passage.fiscal_year,
        accession_number=passage.accession_number,
        rrf_score=0.0,
        semantic_rank=None,
        lexical_rank=None,
        neighbors=(),
    )


def _neighbor_block(
    passage: RetrievedPassage,
    neighbor: NeighborChunk,
    passages: list[RetrievedPassage],
    number: int,
) -> str:
    extra = _year_extra(
        passage.document_id, neighbor.chunk_index, passages, neighbor.content
    )
    if passage.content not in extra:
        extra.append(passage.content)
    body = _with_year_columns(
        neighbor.content,
        extra,
        _header(passage),
    )
    return (
        f"chunk: {number}\n"
        f"ticker: {passage.ticker}\n"
        f"company: {passage.company_name}\n"
        f"filing: {passage.filing_type} FY{passage.fiscal_year} "
        f"filed {passage.filing_date.isoformat()}\n"
        f"section: {section_name(neighbor.content, None)}\n"
        f"{body}"
    )


def _passage_label(passage: RetrievedPassage) -> str:
    return f"{passage.ticker} {passage.filing_type} FY{passage.fiscal_year}"


def _block(
    passage: RetrievedPassage, passages: list[RetrievedPassage], number: int
) -> str:
    page = passage.metadata.get("page")
    section = section_name(passage.content, passage.metadata.get("section"))
    extra = _year_extra(
        passage.document_id, passage.chunk_index, passages, passage.content
    )
    return (
        f"chunk: {number}\n"
        f"ticker: {passage.ticker}\n"
        f"company: {passage.company_name}\n"
        f"filing: {passage.filing_type} FY{passage.fiscal_year} "
        f"filed {passage.filing_date.isoformat()}\n"
        f"page: {page}\n"
        f"section: {section}\n"
        f"{_with_year_columns(passage.content, extra, _header(passage))}"
    )


def _year_extra(
    document_id: UUID,
    chunk_index: int,
    passages: list[RetrievedPassage],
    own_content: str,
) -> list[str]:
    """Nearby chunks of the same filing, so a year header can label a later row."""
    extra: list[str] = []
    seen: set[str] = set()

    def add(content: str) -> None:
        if not content or content == own_content or content in seen:
            return
        seen.add(content)
        extra.append(content)

    for passage in passages:
        if passage.document_id != document_id:
            continue
        if abs(passage.chunk_index - chunk_index) <= YEAR_HEADER_SPAN:
            add(passage.content)
        for neighbor in passage.neighbors:
            if abs(neighbor.chunk_index - chunk_index) > YEAR_HEADER_SPAN:
                continue
            add(neighbor.content)
    return extra


def _header(passage: RetrievedPassage) -> str:
    return filing_header(
        passage.company_name, passage.filing_type, passage.fiscal_year
    )


def _with_year_columns(content: str, extra: list[str], header: str) -> str:
    labels = year_column_lines(content, extra)
    body = f"content:\n{header}\n{content}"
    if not labels:
        return body
    return f"{body}\nYear columns:\n" + "\n".join(labels)
