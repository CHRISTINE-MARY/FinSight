"""Scripted check of the client-brief questions against an ingested corpus.

Run with `pytest -m integration`. The fast suite is `pytest -m "not integration"`.
"""

from __future__ import annotations

import re

import pytest
from openai import RateLimitError
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

# Questions 1–9 name companies. Question 10 only has to surface margin or AI text.
BRIEF_QUESTIONS: list[tuple[str, frozenset[str] | None]] = [
    (
        (
            "Across Apple's 2021–2025 10-Ks, how did the revenue mix between "
            "iPhone, Services, Mac, iPad, and Wearables change, and which "
            "category appears to have contributed most to any mix shift?"
        ),
        frozenset({"AAPL"}),
    ),
    (
        (
            "For Amazon, compare AWS operating income and margin against North "
            "America and International from 2021–2025. In which years did AWS "
            "appear to fund losses or weaker profitability elsewhere?"
        ),
        frozenset({"AMZN"}),
    ),
    (
        (
            "How did NVIDIA describe demand drivers, customer concentration, and "
            "supply constraints for its Data Center business from fiscal 2021 "
            "through fiscal 2025?"
        ),
        frozenset({"NVDA"}),
    ),
    (
        (
            "Across Microsoft's 2021–2025 filings, what changed in the way the "
            "company describes Azure, AI infrastructure, and cloud capacity "
            "constraints?"
        ),
        frozenset({"MSFT"}),
    ),
    (
        (
            "For Alphabet, how did Google Search, YouTube ads, Google Network, "
            "subscriptions/platforms/devices, and Google Cloud revenue trends "
            "differ across the available 10-Ks?"
        ),
        frozenset({"GOOGL"}),
    ),
    (
        (
            "Which of the five companies added, removed, or materially changed "
            "risk-factor language related to AI, cloud infrastructure, export "
            "controls, supply chain concentration, or regulation between 2021 "
            "and 2025?"
        ),
        frozenset({"AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"}),
    ),
    (
        (
            "For Apple and NVIDIA, what do the filings say about supplier "
            "concentration or dependence on third-party manufacturing, and did "
            "the wording become more or less urgent over time?"
        ),
        frozenset({"AAPL", "NVDA"}),
    ),
    (
        (
            "Compare capital expenditures and purchase commitments for "
            "Microsoft, Alphabet, Amazon, and NVIDIA. What do the filings imply "
            "about the scale and timing of AI/cloud infrastructure investment?"
        ),
        frozenset({"MSFT", "GOOGL", "AMZN", "NVDA"}),
    ),
    (
        (
            "For each company, summarize the most important geographic revenue "
            "exposures disclosed in the latest 10-K, then identify any "
            "year-over-year changes that could matter to an analyst."
        ),
        frozenset({"AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"}),
    ),
    (
        (
            "If an analyst asks whether the filings prove that generative AI "
            "improved margins for any of these companies, what evidence exists "
            "in the corpus, and where should the bot refuse to infer beyond "
            "the filings?"
        ),
        None,
    ),
]

_MARGIN_OR_AI = re.compile(r"margin|\bAI\b", re.IGNORECASE)


@pytest.mark.integration
def test_client_brief_questions_return_relevant_chunks() -> None:
    try:
        from app.database.documents import chunk_status_counts
        from app.retrieval.retriever import retrieve
    except ValidationError as exc:
        pytest.skip(f"Settings are not available: {exc}")
    try:
        embedded = chunk_status_counts().get("embedded", 0)
    except SQLAlchemyError as exc:
        pytest.skip(f"Database is not available: {exc}")
    if embedded == 0:
        pytest.skip("No embedded chunks in the corpus")

    for question, tickers in BRIEF_QUESTIONS:
        try:
            passages = retrieve(question)
        except RateLimitError as exc:
            pytest.skip(f"Embedding quota is exhausted: {exc}")
        assert passages, question
        if tickers is None:
            parts = [passage.content for passage in passages]
            parts.extend(
                neighbor.content
                for passage in passages
                for neighbor in passage.neighbors
            )
            assert _MARGIN_OR_AI.search(" ".join(parts)), question
            continue
        found = {passage.ticker for passage in passages}
        assert found & tickers, f"{question} returned {sorted(found)}"
