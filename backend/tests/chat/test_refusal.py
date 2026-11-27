from dataclasses import replace

from app.chat.refusal import Filer, evidence_sufficient, out_of_scope
from tests.chat.test_orchestrator import _passage

FILERS = [
    Filer("AAPL", "Apple Inc."),
    Filer("AMZN", "Amazon.com, Inc."),
    Filer("MSFT", "Microsoft Corporation"),
    Filer("NVDA", "NVIDIA Corporation"),
    Filer("GOOGL", "Alphabet Inc."),
]


def test_a_company_missing_from_filing_metadata_is_out_of_scope() -> None:
    assert out_of_scope("What were Berkshire Hathaway's 2024 operating earnings?", FILERS)
    assert out_of_scope("What was Tesla, Inc.'s automotive revenue?", FILERS)
    assert out_of_scope("What were Walmart's fiscal 2024 net sales?", FILERS)


def test_a_filer_named_in_the_question_is_in_scope() -> None:
    question = (
        "What share of NVIDIA revenue came from customers outside the United States?"
    )

    assert not out_of_scope(question, FILERS)
    assert not out_of_scope("What were Apple's iPad net sales for fiscal 2024?", FILERS)


def test_a_question_with_no_company_is_left_to_the_evidence_check() -> None:
    assert not out_of_scope("How much bitcoin did any company in the corpus hold?", FILERS)
    assert not out_of_scope("What were Mac net sales?", FILERS)


def test_coverage_requires_the_asked_company() -> None:
    apple = "What were Apple's iPhone net sales for fiscal 2024?"
    microsoft = replace(_passage(), ticker="MSFT", company_name="Microsoft Corporation")

    assert evidence_sufficient(apple, [_passage()], FILERS)
    assert not evidence_sufficient(apple, [microsoft], FILERS)


def test_an_unnamed_subject_needs_both_searches_to_agree() -> None:
    question = "How much bitcoin did any company in the corpus hold?"
    agreed = _passage()
    semantic_only = replace(_passage(), lexical_rank=None)

    assert evidence_sufficient(question, [agreed], FILERS)
    assert not evidence_sufficient(question, [semantic_only], FILERS)
    assert not evidence_sufficient(question, [], FILERS)
