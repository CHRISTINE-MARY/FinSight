import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Self
from uuid import UUID

from app.assistant.outputs import AnswerTable, Citation, GroundedAnswer
from app.auth.dependencies import CurrentUser
from app.chat.orchestrator import run_turn
from app.database.chats import StoredMessage, ThreadRecord
from app.retrieval.retriever import RetrievedPassage

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
    def __init__(self, output: GroundedAnswer) -> None:
        self._output = output
        self.prompts: list[str] = []

    def iter(self, prompt: str, *, deps: object = None) -> FakeRun:
        self.prompts.append(prompt)
        return FakeRun(self._output)


def _no_history(monkeypatch) -> None:
    async def list_messages(_user: CurrentUser, _thread_id: UUID) -> list[StoredMessage]:
        return []

    monkeypatch.setattr("app.chat.orchestrator.list_messages", list_messages)


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
    assert turn.citations[0].excerpt == "iPhone"
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


def _message(role: str, text: str) -> StoredMessage:
    return StoredMessage(
        id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        role=role,
        parts=[{"type": "text", "text": text}],
    )
