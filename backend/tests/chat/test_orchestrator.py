import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Self
from uuid import UUID

from app.assistant.outputs import Citation, GroundedAnswer
from app.auth.dependencies import CurrentUser
from app.chat.orchestrator import run_turn
from app.database.chats import ThreadRecord
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

    def iter(self, _prompt: str, *, deps: object = None) -> FakeRun:
        return FakeRun(self._output)


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
        def search(self, _question: str, *, top_k: int = 8, progress=None):
            if progress is not None:
                progress("retrieve", "Embedding the question")
            return [passage]

    output = GroundedAnswer(
        answer="iPhone revenue grew.",
        citations=[Citation(chunk_id=CHUNK_ID, excerpt="iPhone")],
    )
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
        def search(self, _question: str, *, top_k: int = 8, progress=None):
            if progress is not None:
                progress("retrieve", "No corpus match")
            return []

    output = GroundedAnswer(
        answer="The corpus does not contain enough evidence.",
        citations=[],
    )
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
