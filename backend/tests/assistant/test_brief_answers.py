"""Live answers for the client-brief checks. Run with `pytest -m integration`."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID

import pytest
from openai import RateLimitError
from pydantic_ai.exceptions import ModelQuotaExceededError, ModelRateLimitError
from sqlalchemy.exc import SQLAlchemyError

from app.auth.dependencies import CurrentUser
from app.chat.orchestrator import AssistantError, RetrievalError, run_turn
from app.database.chats import ThreadRecord
from app.database.documents import chunk_status_counts
from app.grounding.validator import says_not_enough

USER = CurrentUser(
    id=UUID("6b1f3c2a-7e4d-4a1b-9c8d-0e1f2a3b4c5d"),
    email="analyst@example.com",
    access_token="token",
)
THREAD = ThreadRecord(
    id=UUID("11111111-1111-4111-8111-111111111111"),
    title="Brief checks",
    created_at=datetime(2026, 11, 18, tzinfo=UTC),
    updated_at=datetime(2026, 11, 18, tzinfo=UTC),
)

APPLE_MIX = (
    "Across Apple's 2021-2025 10-Ks, how did the revenue mix between iPhone, "
    "Services, Mac, iPad, and Wearables change?"
)
UNDER_SPECIFIED = "How is the company doing this year?"
AI_MARGINS = (
    "Do the filings prove that generative AI improved margins for any of "
    "Apple, Amazon, Alphabet, Microsoft, or NVIDIA?"
)
_REFUSAL = (
    "not enough evidence",
    "does not contain enough evidence",
    "do not infer",
    "cannot infer",
    "does not prove",
    "does not establish",
    "cannot conclude",
)


@pytest.mark.integration
def test_client_brief_answers_stay_inside_the_filings() -> None:
    try:
        embedded = chunk_status_counts().get("embedded", 0)
    except SQLAlchemyError as exc:
        pytest.skip(f"Database is not available: {exc}")
    if embedded == 0:
        pytest.skip("No embedded chunks in the corpus")

    cited = _ask(APPLE_MIX)
    assert cited.citations
    filing = cited.citations[0].data
    assert filing["ticker"] == "AAPL"
    assert filing["filingType"]
    assert filing["fiscalYear"]
    assert filing["accessionNumber"]

    thin = _ask(UNDER_SPECIFIED)
    assert says_not_enough(thin.answer)

    margins = _ask(AI_MARGINS)
    lowered = margins.answer.lower()
    assert any(phrase in lowered for phrase in _REFUSAL)


def _ask(question: str):
    try:
        return asyncio.run(run_turn(USER, THREAD, question))
    except (AssistantError, RetrievalError) as exc:
        if _quota(exc) or _quota(exc.__cause__):
            pytest.skip(f"Chat model quota is exhausted: {exc.__cause__ or exc}")
        raise


def _quota(exc: BaseException | None) -> bool:
    return isinstance(exc, (ModelRateLimitError, ModelQuotaExceededError, RateLimitError))
