from uuid import UUID

import pytest

from app.retrieval.fusion import reciprocal_rank_fusion

BOTH = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
ONLY_SEMANTIC = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
ONLY_LEXICAL = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
REPEATED = UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd")


def test_chunk_in_both_lists_outranks_a_single_list() -> None:
    fused = reciprocal_rank_fusion([[BOTH, ONLY_SEMANTIC], [BOTH]])
    scores = dict(fused)

    assert scores[BOTH] == pytest.approx(1 / 61 + 1 / 61)
    assert scores[ONLY_SEMANTIC] == pytest.approx(1 / 62)
    assert fused[0][0] == BOTH


def test_order_follows_score_and_ties_keep_first_seen_order() -> None:
    fused = reciprocal_rank_fusion([[ONLY_SEMANTIC], [ONLY_LEXICAL]])

    assert [chunk_id for chunk_id, _score in fused] == [ONLY_SEMANTIC, ONLY_LEXICAL]
    assert fused[0][1] == pytest.approx(fused[1][1])


def test_empty_rankings_return_nothing() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_duplicate_in_one_list_counts_at_its_first_rank() -> None:
    fused = reciprocal_rank_fusion([[REPEATED, REPEATED, ONLY_SEMANTIC]])
    scores = dict(fused)

    assert scores[REPEATED] == pytest.approx(1 / 61)
    assert scores[ONLY_SEMANTIC] == pytest.approx(1 / 63)
