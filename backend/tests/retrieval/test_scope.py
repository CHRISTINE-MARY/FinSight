from uuid import UUID

import pytest

from app.chat.orchestrator import _prompt
from app.retrieval.retriever import SCOPE_BUDGET, retrieve
from app.retrieval.scope import (
    FilingScope,
    filing_header,
    question_scopes,
    scope_query,
)
from tests.retrieval.test_queries import HIT, _Connection, _Engine


def test_one_company_and_year_is_one_filtered_search() -> None:
    scopes = question_scopes(
        "Where does Apple's fiscal 2025 Form 10-K place its principal executive offices?"
    )

    assert scopes == [FilingScope("AAPL", 2025)]


def test_comparison_is_one_scope_per_year() -> None:
    scopes = question_scopes(
        "What full-time equivalent employee counts did Apple report "
        "for fiscal 2022, fiscal 2023, and fiscal 2025, in that order?"
    )

    assert scopes == [
        FilingScope("AAPL", 2022),
        FilingScope("AAPL", 2023),
        FilingScope("AAPL", 2025),
    ]


def test_two_companies_in_one_year_are_separate_searches() -> None:
    scopes = question_scopes(
        "What were Apple and Microsoft net sales in fiscal 2024?"
    )

    assert scopes == [FilingScope("AAPL", 2024), FilingScope("MSFT", 2024)]


def test_a_question_without_a_filer_stays_unfiltered() -> None:
    assert question_scopes("How did iPhone revenue change?") == [
        FilingScope(None, None)
    ]


def test_a_calendar_date_uses_the_company_year_end() -> None:
    apple = question_scopes(
        "How many employees did Apple have on December 31, 2024?"
    )
    microsoft = question_scopes(
        "How many employees did Microsoft have on December 31, 2024?"
    )
    alphabet = question_scopes(
        "What employee counts did Alphabet report at December 31, "
        "2022, 2023, 2024, and 2025, in that order?"
    )
    nvidia = question_scopes(
        "How many employees did NVIDIA have at the end of January 2025?"
    )

    assert apple == [FilingScope("AAPL", 2025)]
    assert microsoft == [FilingScope("MSFT", 2025)]
    assert nvidia == [FilingScope("NVDA", 2025)]
    assert alphabet == [
        FilingScope("GOOGL", 2022),
        FilingScope("GOOGL", 2023),
        FilingScope("GOOGL", 2024),
        FilingScope("GOOGL", 2025),
    ]


def test_june_dates_stay_in_microsofts_fiscal_year() -> None:
    scopes = question_scopes(
        "How many people did Microsoft employ full-time on June 30, 2024 "
        "and on June 30, 2025, in that order?"
    )

    assert scopes == [FilingScope("MSFT", 2024), FilingScope("MSFT", 2025)]


def test_employee_subquery_names_the_year_end() -> None:
    query = scope_query(
        "How many employees did Apple report for fiscal 2025?",
        FilingScope("AAPL", 2025),
    )

    assert query.startswith(
        "full-time employees as of the last Saturday of September 2025. "
        "human capital."
    )


def test_an_unavailable_filing_is_named_in_the_prompt() -> None:
    text = _prompt(
        "How many employees did Apple report for fiscal 2026?",
        [],
        [],
        {},
        ["AAPL FY2026 is not in the corpus."],
    )

    assert "AAPL FY2026 is not in the corpus." in text
    assert "Do not invent a figure for it." in text


def test_filing_header_uses_the_short_company_name() -> None:
    assert filing_header("Apple Inc.", "10-K", 2023) == "Apple 10-K FY2023,"
    assert filing_header("Amazon.com, Inc.", "10-K", 2024) == "Amazon 10-K FY2024,"
    assert (
        filing_header("Microsoft Corporation", "10-K", 2025)
        == "Microsoft 10-K FY2025,"
    )


def _install_search(monkeypatch: pytest.MonkeyPatch, semantic) -> list[str]:
    queries: list[str] = []

    def lexical(_connection, query, _limit, **_kwargs):
        queries.append(query)
        return []

    monkeypatch.setattr(
        "app.retrieval.retriever.embed_texts",
        lambda texts, **_kwargs: [[0.1] for _ in texts],
    )
    monkeypatch.setattr("app.retrieval.retriever.semantic_search", semantic)
    monkeypatch.setattr("app.retrieval.retriever.full_text_search", lexical)
    monkeypatch.setattr(
        "app.retrieval.retriever.database_engine", lambda: _Engine(_Connection())
    )
    monkeypatch.setattr(
        "app.retrieval.retriever.indexed_fiscal_years",
        lambda _connection, _ticker: set(range(2021, 2026)),
    )
    monkeypatch.setattr(
        "app.retrieval.retriever.load_passages",
        lambda _connection, hits: [hit.chunk_id for hit in hits],
    )
    return queries


def test_each_year_keeps_its_own_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str | None, int | None]] = []

    def semantic(_connection, _embedding, _limit, *, ticker, fiscal_year):
        calls.append((ticker, fiscal_year))
        pool = {
            2022: [UUID(int=index) for index in range(1, 8)],
            2023: [UUID(int=index) for index in range(1, 8)],
            2025: [UUID(int=index) for index in range(20, 27)],
        }
        return pool[fiscal_year]

    queries = _install_search(monkeypatch, semantic)
    question = (
        "What employee counts did Apple report for fiscal 2022, "
        "fiscal 2023, and fiscal 2025?"
    )
    loaded = retrieve(question, top_k=8, scope_k=SCOPE_BUDGET, candidate_k=50)

    assert calls == [("AAPL", 2022), ("AAPL", 2023), ("AAPL", 2025)]
    # 2023 repeats 2022's first chunks, so only its two new chunks are added.
    assert loaded[:SCOPE_BUDGET] == [UUID(int=index) for index in range(1, 6)]
    assert loaded[SCOPE_BUDGET : SCOPE_BUDGET + 2] == [
        UUID(int=index) for index in range(6, 8)
    ]
    assert loaded[SCOPE_BUDGET + 2 :] == [UUID(int=index) for index in range(20, 25)]
    assert all("full-time employees as of" in query for query in queries)
    assert all("human capital" in query for query in queries)
    assert "September 2022" in queries[0]
    assert "September 2025" in queries[2]


def test_each_company_keeps_its_own_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    def semantic(_connection, _embedding, _limit, *, ticker, fiscal_year):
        start = 1 if ticker == "AAPL" else 30
        return [UUID(int=index) for index in range(start, start + 6)]

    _install_search(monkeypatch, semantic)
    loaded = retrieve(
        "What were Apple and Microsoft full-time employees in fiscal 2024?",
        scope_k=5,
        candidate_k=50,
    )

    assert loaded[:5] == [UUID(int=index) for index in range(1, 6)]
    assert loaded[5:] == [UUID(int=index) for index in range(30, 35)]


def test_a_missing_fiscal_year_is_not_searched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int | None] = []

    def semantic(_connection, _embedding, _limit, *, ticker, fiscal_year):
        calls.append(fiscal_year)
        return [HIT]

    _install_search(monkeypatch, semantic)
    monkeypatch.setattr(
        "app.retrieval.retriever.indexed_fiscal_years",
        lambda _connection, _ticker: {2024},
    )
    gaps: list[str] = []
    loaded = retrieve(
        "How many employees did Apple report for fiscal 2024 and fiscal 2025?",
        unavailable=gaps,
    )

    assert calls == [2024]
    assert loaded == [HIT]
    assert gaps == ["AAPL FY2025 is not in the corpus."]


def test_candidates_rerank_down_to_eight(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    def semantic(_connection, _embedding, limit, *, ticker, fiscal_year):
        seen.append(limit)
        return [UUID(int=index) for index in range(1, limit + 1)]

    _install_search(monkeypatch, semantic)

    hits = retrieve("How many employees did Apple have in fiscal 2024?")

    assert seen == [50]
    assert len(hits) == 8
