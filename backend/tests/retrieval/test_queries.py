from datetime import date
from typing import Self
from uuid import UUID

import pytest

from app.retrieval.queries import (
    FETCH_SURROUNDING,
    RankedHit,
    full_text_search,
    load_passages,
    semantic_search,
)
from app.retrieval.retriever import retrieve

HIT = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
SECOND = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
THIRD = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
DOCUMENT = UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd")
BEFORE = UUID("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee")
AFTER = UUID("ffffffff-ffff-4fff-8fff-ffffffffffff")
FILED = date(2024, 11, 1)


class _Result:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows


class _Connection:
    def __init__(self, rows: list[object] | None = None) -> None:
        self.statement = None
        self.params = None
        self._rows = rows or []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def execute(self, statement: object, params: dict[str, object]) -> _Result:
        self.statement = statement
        self.params = params
        return _Result(self._rows)


class _Row:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


def test_semantic_search_uses_cosine_distance_and_embedded_chunks() -> None:
    connection = _Connection()
    vector = [0.25, 0.5]

    assert semantic_search(connection, vector, 50) == []

    sql = str(connection.statement)
    assert "<=>" in sql
    assert "embedding_status = 'embedded'" in sql
    assert connection.params == {"query_vector": "[0.25,0.5]", "limit": 50}


def test_surrounding_query_selects_neighbor_indexes() -> None:
    sql = str(FETCH_SURROUNDING)
    assert "chunk_index - 1" in sql
    assert "chunk.id <> :chunk_id" in sql


def test_full_text_search_ranks_the_search_vector() -> None:
    connection = _Connection()
    question = "How did iPhone revenue change?"

    assert full_text_search(connection, question, 50) == []

    sql = str(connection.statement)
    assert "search_vector" in sql
    assert "ts_rank_cd" in sql
    assert connection.params == {"query": question, "limit": 50}


def test_load_passages_keeps_neighbors_out_of_the_hit() -> None:
    connection = _Connection(
        [
            _passage_row(BEFORE, HIT, chunk_index=3, content="previous"),
            _passage_row(HIT, HIT, chunk_index=4, content="iPhone revenue"),
            _passage_row(AFTER, HIT, chunk_index=5, content="next"),
        ]
    )
    hit = RankedHit(chunk_id=HIT, rrf_score=0.03, semantic_rank=1, lexical_rank=2)

    passages = load_passages(connection, [hit])

    assert "chunk_index - 1" in str(connection.statement)
    assert "chunk_index + 1" in str(connection.statement)
    assert connection.params == {"chunk_ids": [HIT]}
    assert len(passages) == 1
    passage = passages[0]
    assert passage.chunk_id == HIT
    assert passage.content == "iPhone revenue"
    assert passage.ticker == "AAPL"
    assert passage.rrf_score == 0.03
    assert passage.semantic_rank == 1
    assert passage.lexical_rank == 2
    assert [neighbor.chunk_index for neighbor in passage.neighbors] == [3, 5]
    assert [neighbor.content for neighbor in passage.neighbors] == ["previous", "next"]


def test_retrieve_loads_only_the_fused_top_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _Connection(
        [
            _passage_row(BEFORE, HIT, chunk_index=3, content="previous"),
            _passage_row(HIT, HIT, chunk_index=4, content="iPhone mix"),
            _passage_row(AFTER, HIT, chunk_index=5, content="next"),
            _passage_row(SECOND, SECOND, chunk_index=1, content="Services"),
        ]
    )
    monkeypatch.setattr(
        "app.retrieval.retriever.embed_texts", lambda _texts: [[0.1, 0.2]]
    )
    monkeypatch.setattr(
        "app.retrieval.retriever.semantic_search",
        lambda _connection, _embedding, limit: [HIT, SECOND] if limit == 50 else [],
    )
    monkeypatch.setattr(
        "app.retrieval.retriever.full_text_search",
        lambda _connection, _query, _limit: [HIT, THIRD],
    )
    monkeypatch.setattr(
        "app.retrieval.retriever.database_engine",
        lambda: _Engine(connection),
    )

    passages = retrieve("How did Apple's iPhone mix change?", top_k=2, candidate_k=50)

    assert [passage.chunk_id for passage in passages] == [HIT, SECOND]
    assert connection.params == {"chunk_ids": [HIT, SECOND]}
    assert [neighbor.chunk_index for neighbor in passages[0].neighbors] == [3, 5]
    assert passages[0].semantic_rank == 1
    assert passages[0].lexical_rank == 1
    assert passages[1].semantic_rank == 2
    assert passages[1].lexical_rank is None


def test_retrieve_returns_empty_when_both_searches_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.retrieval.retriever.embed_texts", lambda _texts: [[0.1, 0.2]]
    )
    monkeypatch.setattr(
        "app.retrieval.retriever.semantic_search",
        lambda _connection, _embedding, _limit: [],
    )
    monkeypatch.setattr(
        "app.retrieval.retriever.full_text_search",
        lambda _connection, _query, _limit: [],
    )
    monkeypatch.setattr(
        "app.retrieval.retriever.database_engine",
        lambda: _Engine(_Connection()),
    )
    monkeypatch.setattr(
        "app.retrieval.retriever.load_passages",
        lambda _connection, _hits: pytest.fail("load_passages should not run"),
    )

    assert retrieve("no matching filing") == []


def test_retrieve_rejects_a_blank_question() -> None:
    with pytest.raises(ValueError, match="empty"):
        retrieve("   ")


class _Engine:
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection

    def connect(self) -> _Connection:
        return self._connection


def _passage_row(
    chunk_id: UUID, hit_id: UUID, *, chunk_index: int, content: str
) -> _Row:
    return _Row(
        hit_id=hit_id,
        id=chunk_id,
        document_id=DOCUMENT,
        chunk_index=chunk_index,
        content=content,
        token_count=12,
        metadata={"page": 20},
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type="10-K",
        filing_date=FILED,
        fiscal_year=2024,
        accession_number="0000320193-24-000123",
    )
