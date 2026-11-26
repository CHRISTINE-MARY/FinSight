from app.retrieval.retriever import _ranked_lists, _search_text

LATEST = "Give me 2022 sales"
EARLIER = "give me apples 2021 sales"


def test_follow_up_repeats_the_current_question_and_keeps_earlier_ones() -> None:
    text = _search_text(LATEST, [EARLIER])

    assert text == f"{LATEST}\n{LATEST}\n{EARLIER}"
    assert text.count(LATEST) > text.count(EARLIER)


def test_search_text_without_history_is_the_question() -> None:
    assert _search_text(LATEST, []) == LATEST


def test_context_hits_join_the_latest_question_lists() -> None:
    semantic = ["semantic"]
    lexical = ["lexical"]
    context = ["context"]

    assert _ranked_lists(semantic, lexical, context) == [semantic, lexical, context]
    assert _ranked_lists(semantic, lexical, []) == [semantic, lexical]
