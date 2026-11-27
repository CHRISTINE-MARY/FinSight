import pytest
from pydantic import ValidationError

from app.assistant.outputs import (
    REFUSAL_SENTENCE,
    AnswerTable,
    Claim,
    ModelAnswer,
    compose_answer,
)


def test_compose_answer_appends_a_markdown_table() -> None:
    text = compose_answer(
        "Services gained share.",
        AnswerTable(
            headers=["Segment", "FY2024"],
            rows=[["iPhone | Wearables", "51%"], ["Services"]],
        ),
    )
    assert text == (
        "Services gained share.\n\n"
        "| Segment | FY2024 |\n"
        "| --- | --- |\n"
        "| iPhone / Wearables | 51% |\n"
        "| Services |  |"
    )


def test_compose_answer_leaves_a_paragraph_unchanged() -> None:
    assert compose_answer("Services gained share.", None) == "Services gained share."


def test_an_empty_model_answer_is_rejected() -> None:
    with pytest.raises(ValidationError, match="empty object"):
        ModelAnswer()
    with pytest.raises(ValidationError, match="empty object"):
        ModelAnswer.model_validate({})


def test_a_refusal_or_a_claim_is_accepted() -> None:
    refused = ModelAnswer(answer=REFUSAL_SENTENCE)
    claimed = ModelAnswer(claims=[Claim(text="Apple had 166,000 employees.", chunk=1)])

    assert refused.claims == []
    assert claimed.answer == ""


def test_compose_answer_does_not_duplicate_an_existing_table() -> None:
    existing = "Lead.\n\n| Segment | FY2024 |\n| --- | --- |\n| iPhone | 51% |"
    assert (
        compose_answer(
            existing,
            AnswerTable(headers=["Segment", "FY2024"], rows=[["Services", "25%"]]),
        )
        == existing
    )
