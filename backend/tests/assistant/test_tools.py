import asyncio
from datetime import date
from uuid import UUID

from app.assistant.agent import read_chunk, search_filings
from app.assistant.deps import DocumentAgentDeps
from app.retrieval.retriever import RetrievedPassage

CHUNK_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


def test_search_filings_records_passages_for_grounding() -> None:
    passage = _passage()
    deps = DocumentAgentDeps(
        user_id="user",
        thread_id="thread",
        retriever=_Retriever(passage),
    )

    rows = asyncio.run(search_filings(_Context(deps), "iPhone revenue"))

    assert deps.retrieved[CHUNK_ID] is passage
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["chunk"] == 1
    assert "chunk_id" not in rows[0]


def test_read_chunk_rejects_an_unknown_id() -> None:
    deps = DocumentAgentDeps(
        user_id="user",
        thread_id="thread",
        retriever=_Retriever(_passage()),
    )

    result = asyncio.run(read_chunk(_Context(deps), 9))

    assert result == {"error": "Unknown passage number"}
    assert deps.retrieved == {}


class _Context:
    def __init__(self, deps: DocumentAgentDeps) -> None:
        self.deps = deps


class _Retriever:
    def __init__(self, passage: RetrievedPassage) -> None:
        self.passage = passage

    def search(
        self, _question: str, *, top_k: int = 8, progress: object = None
    ) -> list[RetrievedPassage]:
        assert top_k == 8
        assert progress is None
        return [self.passage]

    def read_chunk(self, chunk_id: UUID) -> RetrievedPassage | None:
        if chunk_id == self.passage.chunk_id:
            return self.passage
        return None

    def read_surrounding(self, _chunk_id: UUID) -> list[RetrievedPassage]:
        return []


def _passage() -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=CHUNK_ID,
        document_id=UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc"),
        chunk_index=4,
        content="iPhone net sales were $201,183 million.",
        token_count=12,
        metadata={"page": [42]},
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type="10-K",
        filing_date=date(2024, 11, 1),
        fiscal_year=2024,
        accession_number="0000320193-24-000123",
        rrf_score=0.0,
        semantic_rank=None,
        lexical_rank=None,
        neighbors=(),
    )
