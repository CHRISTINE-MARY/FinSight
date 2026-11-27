import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Self
from uuid import UUID

import pytest

from app.assistant.outputs import (
    AnswerTable,
    Citation,
    Claim,
    GroundedAnswer,
    ModelAnswer,
)
from app.auth.dependencies import CurrentUser
from app.chat.orchestrator import run_turn
from app.database.chats import StoredMessage, ThreadRecord
from app.grounding.tables import GroundingError
from app.retrieval.retriever import NeighborChunk, RetrievedPassage

USER = CurrentUser(
    id=UUID("6b1f3c2a-7e4d-4a1b-9c8d-0e1f2a3b4c5d"),
    email="analyst@example.com",
    access_token="token",
)
THREAD = ThreadRecord(
    id=UUID("11111111-1111-4111-8111-111111111111"),
    title="New chat",
    created_at=datetime(2026, 11, 11, tzinfo=UTC),
    updated_at=datetime(2026, 11, 11, tzinfo=UTC),
)
CHUNK_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


class ModelRequestNode:
    pass


class CallToolsNode:
    def __init__(self) -> None:
        self.model_response = SimpleNamespace(
            parts=[SimpleNamespace(part_kind="tool-call", tool_name="search_filings")]
        )


class FakeRun:
    def __init__(self, output: GroundedAnswer) -> None:
        self.result = SimpleNamespace(output=output)
        self._nodes: list[object] = [ModelRequestNode(), CallToolsNode()]
        self._index = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> object:
        if self._index >= len(self._nodes):
            raise StopAsyncIteration
        node = self._nodes[self._index]
        self._index += 1
        return node


class FakeAgent:
    def __init__(self, output: object) -> None:
        self._outputs = output if isinstance(output, list) else [output]
        self.prompts: list[str] = []
        self._calls = 0

    def iter(self, prompt: str, *, deps: object = None) -> FakeRun:
        self.prompts.append(prompt)
        chosen = self._outputs[min(self._calls, len(self._outputs) - 1)]
        self._calls += 1
        return FakeRun(chosen)


def _no_history(monkeypatch) -> None:
    async def list_messages(_user: CurrentUser, _thread_id: UUID) -> list[StoredMessage]:
        return []

    monkeypatch.setattr("app.chat.orchestrator.list_messages", list_messages)
    _corpus(monkeypatch)


def _corpus(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.chat.orchestrator.list_filers",
        lambda: [
            ("AAPL", "Apple Inc."),
            ("AMZN", "Amazon.com, Inc."),
            ("MSFT", "Microsoft Corporation"),
            ("NVDA", "NVIDIA Corporation"),
            ("GOOGL", "Alphabet Inc."),
        ],
    )


def _passage() -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=CHUNK_ID,
        document_id=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
        chunk_index=0,
        content="iPhone revenue grew.",
        token_count=4,
        metadata={"page": "42", "section": "Products"},
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type="10-K",
        filing_date=datetime(2024, 11, 1, tzinfo=UTC).date(),
        fiscal_year=2024,
        accession_number="0000320193-24-000123",
        rrf_score=0.1,
        semantic_rank=1,
        lexical_rank=1,
        neighbors=(),
    )


def test_run_turn_reports_each_pipeline_phase(monkeypatch) -> None:
    passage = _passage()

    class Retriever:
        def search(
            self,
            _question: str,
            *,
            top_k: int = 8,
            prior_questions: list[str] | None = None,
            progress=None,
        ):
            if progress is not None:
                progress("retrieve", "Embedding the question")
            return [passage]

    output = GroundedAnswer(
        answer="iPhone revenue grew.",
        citations=[Citation(chunk_id=CHUNK_ID, excerpt="iPhone")],
    )
    _no_history(monkeypatch)
    monkeypatch.setattr("app.chat.orchestrator.DocumentRetriever", Retriever)
    monkeypatch.setattr("app.chat.orchestrator.get_agent", lambda: FakeAgent(output))

    async def scenario():
        seen: list[str] = []

        async def report(_phase: str, detail: str) -> None:
            seen.append(detail)

        turn = await run_turn(
            USER, THREAD, "How did iPhone revenue change?", report=report
        )
        return turn, seen

    turn, seen = asyncio.run(scenario())

    assert "Embedding the question" in seen
    assert "Retrieved 1 passage: AAPL 10-K FY2024" in seen
    assert "Model round 1: waiting for the model" in seen
    assert "Model round 1: calling search_filings" in seen
    assert "Checking that every citation maps to a retrieved passage" in seen
    assert "Grounded with 1 citation" in seen
    assert turn.citations[0].excerpt == "iPhone revenue grew."
    citation = next(part for part in turn.parts if part["type"] == "data-citation")
    assert citation["data"]["ticker"] == "AAPL"
    assert citation["data"]["content"] == "iPhone revenue grew."


def test_run_turn_reports_when_nothing_matches(monkeypatch) -> None:
    class Retriever:
        def search(
            self,
            _question: str,
            *,
            top_k: int = 8,
            prior_questions: list[str] | None = None,
            progress=None,
        ):
            if progress is not None:
                progress("retrieve", "No corpus match")
            return []

    output = GroundedAnswer(
        answer="The corpus does not contain enough evidence.",
        citations=[],
    )
    _no_history(monkeypatch)
    monkeypatch.setattr("app.chat.orchestrator.DocumentRetriever", Retriever)
    monkeypatch.setattr("app.chat.orchestrator.get_agent", lambda: FakeAgent(output))

    async def scenario():
        seen: list[str] = []

        async def report(_phase: str, detail: str) -> None:
            seen.append(detail)

        turn = await run_turn(
            USER, THREAD, "What about a private company?", report=report
        )
        return turn, seen

    turn, seen = asyncio.run(scenario())

    assert "No corpus match" in seen
    assert turn.citations == ()
    assert not any(part["type"] == "data-citation" for part in turn.parts)


def test_run_turn_renders_a_comparison_table(monkeypatch) -> None:
    passage = replace(
        _passage(),
        content="iPhone was 52% and then 51%. Services were 22% and then 25%.",
    )

    class Retriever:
        def search(
            self,
            _question: str,
            *,
            top_k: int = 8,
            prior_questions: list[str] | None = None,
            progress=None,
        ):
            return [passage]

    output = GroundedAnswer(
        answer="Services gained share.",
        table=AnswerTable(
            headers=["Segment", "FY2023", "FY2024"],
            rows=[["iPhone", "52%", "51%"], ["Services", "22%", "25%"]],
        ),
        citations=[Citation(chunk_id=CHUNK_ID, excerpt="iPhone")],
    )
    _no_history(monkeypatch)
    monkeypatch.setattr("app.chat.orchestrator.DocumentRetriever", Retriever)
    monkeypatch.setattr("app.chat.orchestrator.get_agent", lambda: FakeAgent(output))

    turn = asyncio.run(run_turn(USER, THREAD, "Show Apple's mix as a table"))

    assert turn.answer.startswith("Services gained share.")
    assert "| Segment | FY2023 | FY2024 |" in turn.answer
    assert "| --- | --- | --- |" in turn.answer
    assert "| iPhone | 52% | 51% |" in turn.answer
    assert "| Services | 22% | 25% |" in turn.answer


def test_thread_question_names_the_company_from_earlier_questions(monkeypatch) -> None:
    called = {"search": False}

    class Retriever:
        def search(self, _question: str, *, top_k: int = 8, prior_questions=None, progress=None):
            called["search"] = True
            return []

    async def history(_user: CurrentUser, _thread_id: UUID) -> list[StoredMessage]:
        return [
            _message("user", "give me apples 2021 sales"),
            _message("assistant", "Amazon's 2022 total net sales were $514 billion."),
            _message("user", "Give me 2022 sales"),
        ]

    monkeypatch.setattr("app.chat.orchestrator.list_messages", history)
    monkeypatch.setattr("app.chat.orchestrator.DocumentRetriever", Retriever)

    company = asyncio.run(
        run_turn(USER, THREAD, "in prev question which company was i talking about")
    )
    choice = asyncio.run(run_turn(USER, THREAD, "wait amazon, apple or microsoft"))

    assert called["search"] is False
    assert company.answer == "Your earlier questions were about Apple."
    assert company.citations == ()
    assert choice.answer == "Your earlier questions were about Apple."


def test_run_turn_passes_recent_user_questions(monkeypatch) -> None:
    passage = _passage()
    seen: dict[str, list[str] | None] = {"prior": None}

    class Retriever:
        def search(
            self,
            _question: str,
            *,
            top_k: int = 8,
            prior_questions: list[str] | None = None,
            progress=None,
        ):
            seen["prior"] = prior_questions
            return [passage]

    output = GroundedAnswer(
        answer="iPhone revenue grew.",
        citations=[Citation(chunk_id=CHUNK_ID, excerpt="iPhone")],
    )
    agent = FakeAgent(output)

    async def history(_user: CurrentUser, _thread_id: UUID) -> list[StoredMessage]:
        messages = [
            _message("user", f"question {index}") for index in range(1, 8)
        ]
        messages.insert(3, _message("assistant", "Services were 22% of sales."))
        return messages

    monkeypatch.setattr("app.chat.orchestrator.list_messages", history)
    _corpus(monkeypatch)
    monkeypatch.setattr("app.chat.orchestrator.DocumentRetriever", Retriever)
    monkeypatch.setattr("app.chat.orchestrator.get_agent", lambda: agent)

    asyncio.run(run_turn(USER, THREAD, "What about the year before?"))

    assert seen["prior"] == [f"question {index}" for index in range(2, 8)]
    prompt = agent.prompts[0]
    assert "The analyst already asked:" in prompt
    assert "- question 2" in prompt
    assert "- question 7" in prompt
    assert "question 1" not in prompt
    assert "Services were 22% of sales." not in prompt


def test_run_turn_sends_adjacent_chunks_and_accepts_their_citations(monkeypatch) -> None:
    previous_id = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
    next_id = UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd")
    passage = _passage()
    passage = RetrievedPassage(
        chunk_id=passage.chunk_id,
        document_id=passage.document_id,
        chunk_index=80,
        content=passage.content,
        token_count=passage.token_count,
        metadata=passage.metadata,
        ticker=passage.ticker,
        company_name=passage.company_name,
        filing_type=passage.filing_type,
        filing_date=passage.filing_date,
        fiscal_year=passage.fiscal_year,
        accession_number=passage.accession_number,
        rrf_score=passage.rrf_score,
        semantic_rank=passage.semantic_rank,
        lexical_rank=passage.lexical_rank,
        neighbors=(
            NeighborChunk(previous_id, 79, "Japan net sales increased."),
            NeighborChunk(next_id, 81, "Mac net sales were 29,984."),
        ),
    )

    class Retriever:
        def search(self, _question: str, *, top_k: int = 8, prior_questions=None, progress=None):
            return [passage]

    output = GroundedAnswer(
        answer="Mac net sales were 29,984.",
        citations=[Citation(chunk_id=next_id, excerpt="29,984")],
    )
    agent = FakeAgent(output)
    _no_history(monkeypatch)
    monkeypatch.setattr("app.chat.orchestrator.DocumentRetriever", Retriever)
    monkeypatch.setattr("app.chat.orchestrator.get_agent", lambda: agent)

    turn = asyncio.run(run_turn(USER, THREAD, "What were Mac net sales?"))

    prompt = agent.prompts[0]
    assert "Japan net sales increased." in prompt
    assert "Mac net sales were 29,984." in prompt
    assert "chunk: 3" in prompt
    assert str(next_id) not in prompt
    assert str(previous_id) not in prompt
    assert turn.citations[0].chunk_id == next_id


def test_prompt_labels_the_year_column_from_the_neighbor_header(monkeypatch) -> None:
    header_id = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
    header = "|  |  | 2024 | 2024 | Change | 2023 | 2023 |"
    row = (
        "| iPhone | iPhone | $ | 201,183 | (2) | $ | 200,583 |\n"
        "| iPad | iPad | 26,694 |  26,694 | (6) | 28,300 |  28,300 |"
    )
    passage = replace(
        _passage(),
        content=row,
        chunk_index=81,
        neighbors=(NeighborChunk(header_id, 80, header),),
    )

    class Retriever:
        def search(self, _question: str, *, top_k: int = 8, prior_questions=None, progress=None):
            return [passage]

    output = GroundedAnswer(
        answer="iPad net sales were 26,694.",
        citations=[Citation(chunk_id=passage.chunk_id, figure="26,694", label="iPad")],
    )
    agent = FakeAgent(output)
    _no_history(monkeypatch)
    monkeypatch.setattr("app.chat.orchestrator.DocumentRetriever", Retriever)
    monkeypatch.setattr("app.chat.orchestrator.get_agent", lambda: agent)

    asyncio.run(run_turn(USER, THREAD, "What were Apple's iPad net sales for fiscal 2024?"))

    prompt = agent.prompts[0]
    assert row in prompt
    assert "iPad 2024: 26,694" in prompt
    assert "iPad 2023: 28,300" in prompt
    assert "iPhone 2024: 201,183" in prompt
    assert "iPhone 2023: 200,583" in prompt
    assert "iPad 2024: $" not in prompt


def test_prompt_labels_a_row_when_the_header_is_inside_the_previous_chunk(
    monkeypatch,
) -> None:
    header_id = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
    header = "|  |  | 2024 | 2024 | Change | 2023 | 2023 |"
    previous = (
        "net sales by category for 2024, 2023 and 2022 (dollars in millions):\n"
        f"{header}\n"
        "| iPad | iPad | 26,694 |  26,694 | (6) | 28,300 |  28,300 |"
    )
    row = "| Wearables | Wearables | 37,005 |  37,005 | (7) | 39,845 |  39,845 |"
    passage = replace(
        _passage(),
        content=row,
        chunk_index=82,
        neighbors=(NeighborChunk(header_id, 81, previous),),
    )

    class Retriever:
        def search(self, _question, *, top_k: int = 8, prior_questions=None, progress=None):
            return [passage]

    output = GroundedAnswer(
        answer="Wearables net sales were 37,005.",
        citations=[
            Citation(chunk_id=passage.chunk_id, figure="37,005", label="Wearables")
        ],
    )
    agent = FakeAgent(output)
    _no_history(monkeypatch)
    monkeypatch.setattr("app.chat.orchestrator.DocumentRetriever", Retriever)
    monkeypatch.setattr("app.chat.orchestrator.get_agent", lambda: agent)

    asyncio.run(
        run_turn(USER, THREAD, "What were Apple's Wearables net sales for fiscal 2024?")
    )

    prompt = agent.prompts[0]
    assert row in prompt
    assert "chunk: 1" in prompt
    assert "chunk: 2" in prompt
    assert "Apple 10-K FY2024," in prompt
    assert str(passage.chunk_id) not in prompt
    assert str(header_id) not in prompt
    assert "Wearables 2024: 37,005" in prompt
    assert "Wearables 2023: 39,845" in prompt


def test_a_missing_citation_is_retried_once_with_the_claim(monkeypatch) -> None:
    rejected = GroundedAnswer(answer="iPhone revenue grew.", citations=[])
    accepted = GroundedAnswer(
        answer="iPhone revenue grew.",
        citations=[Citation(chunk_id=CHUNK_ID, excerpt="iPhone")],
    )
    agent = FakeAgent([rejected, accepted])
    _no_history(monkeypatch)
    monkeypatch.setattr(
        "app.chat.orchestrator.DocumentRetriever", _static_retriever([_passage()])
    )
    monkeypatch.setattr("app.chat.orchestrator.get_agent", lambda: agent)

    turn = asyncio.run(run_turn(USER, THREAD, "How did iPhone revenue change?"))

    assert len(agent.prompts) == 2
    assert "Claim: iPhone revenue grew." in agent.prompts[1]
    assert "The answer has no citation" in agent.prompts[1]
    assert turn.citations[0].chunk_id == CHUNK_ID


def test_an_unseen_chunk_is_retried_once(monkeypatch) -> None:
    unseen = UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd")
    rejected = GroundedAnswer(
        answer="iPhone revenue grew.",
        citations=[Citation(chunk_id=unseen, excerpt="iPhone")],
    )
    accepted = ModelAnswer(
        claims=[Claim(text="iPhone revenue grew.", chunk=1)]
    )
    agent = FakeAgent([rejected, accepted])
    _no_history(monkeypatch)
    monkeypatch.setattr(
        "app.chat.orchestrator.DocumentRetriever", _static_retriever([_passage()])
    )
    monkeypatch.setattr("app.chat.orchestrator.get_agent", lambda: agent)

    turn = asyncio.run(run_turn(USER, THREAD, "How did iPhone revenue change?"))

    assert "was not retrieved" in agent.prompts[1]
    assert turn.answer == "iPhone revenue grew."
    assert turn.citations[0].chunk_id == CHUNK_ID


def test_a_second_grounding_failure_is_not_retried_again(monkeypatch) -> None:
    agent = FakeAgent(GroundedAnswer(answer="iPhone revenue grew."))
    _no_history(monkeypatch)
    monkeypatch.setattr(
        "app.chat.orchestrator.DocumentRetriever", _static_retriever([_passage()])
    )
    monkeypatch.setattr("app.chat.orchestrator.get_agent", lambda: agent)

    with pytest.raises(GroundingError, match="no citation"):
        asyncio.run(run_turn(USER, THREAD, "How did iPhone revenue change?"))

    assert len(agent.prompts) == 2


def test_question_about_another_company_is_refused(monkeypatch) -> None:
    agent = FakeAgent(
        GroundedAnswer(
            answer="Berkshire Hathaway's operating earnings were $80.0 billion.",
            citations=[Citation(chunk_id=CHUNK_ID, figure="80.0")],
        )
    )
    _no_history(monkeypatch)
    monkeypatch.setattr(
        "app.chat.orchestrator.DocumentRetriever",
        _static_retriever([_passage()]),
    )
    monkeypatch.setattr("app.chat.orchestrator.get_agent", lambda: agent)

    turn = asyncio.run(
        run_turn(USER, THREAD, "What were Berkshire Hathaway's 2024 operating earnings?")
    )

    assert turn.answer == "The corpus does not contain enough evidence."
    assert turn.citations == ()
    assert agent.prompts == []


def test_uncorroborated_retrieval_is_refused(monkeypatch) -> None:
    agent = FakeAgent(
        GroundedAnswer(
            answer="NVIDIA held $1,536.28 in bitcoin.",
            citations=[Citation(chunk_id=CHUNK_ID, figure="1,536.28")],
        )
    )
    passage = replace(_passage(), lexical_rank=None)
    _no_history(monkeypatch)
    monkeypatch.setattr(
        "app.chat.orchestrator.DocumentRetriever",
        _static_retriever([passage]),
    )
    monkeypatch.setattr("app.chat.orchestrator.get_agent", lambda: agent)

    turn = asyncio.run(
        run_turn(USER, THREAD, "How much bitcoin did any company in the corpus hold?")
    )

    assert turn.answer == "The corpus does not contain enough evidence."
    assert turn.citations == ()
    assert agent.prompts == []


def test_wrong_company_in_the_retrieved_filings_is_refused(monkeypatch) -> None:
    agent = FakeAgent(
        GroundedAnswer(
            answer="Apple iPhone net sales were 201,183.",
            citations=[Citation(chunk_id=CHUNK_ID, figure="201,183")],
        )
    )
    passage = replace(
        _passage(), ticker="MSFT", company_name="Microsoft Corporation"
    )
    _no_history(monkeypatch)
    monkeypatch.setattr(
        "app.chat.orchestrator.DocumentRetriever",
        _static_retriever([passage]),
    )
    monkeypatch.setattr("app.chat.orchestrator.get_agent", lambda: agent)

    turn = asyncio.run(
        run_turn(USER, THREAD, "What were Apple's iPhone net sales for fiscal 2024?")
    )

    assert turn.answer == "The corpus does not contain enough evidence."
    assert turn.citations == ()
    assert agent.prompts == []


def _static_retriever(passages: list[RetrievedPassage]):
    class Retriever:
        def search(self, _question, *, top_k: int = 8, prior_questions=None, progress=None):
            return passages

    return Retriever


def _message(role: str, text: str) -> StoredMessage:
    return StoredMessage(
        id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        role=role,
        parts=[{"type": "text", "text": text}],
    )
