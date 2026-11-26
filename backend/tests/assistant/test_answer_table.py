from app.assistant.outputs import AnswerTable, compose_answer


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


def test_compose_answer_does_not_duplicate_an_existing_table() -> None:
    existing = "Lead.\n\n| Segment | FY2024 |\n| --- | --- |\n| iPhone | 51% |"
    assert (
        compose_answer(
            existing,
            AnswerTable(headers=["Segment", "FY2024"], rows=[["Services", "25%"]]),
        )
        == existing
    )
