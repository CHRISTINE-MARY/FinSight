"""Reciprocal Rank Fusion.

Dense cosine scores and full-text ranks are not on the same scale, so the
lists are fused by rank. k=60 is the Cormack constant from the 2009 paper.
"""

from __future__ import annotations

from uuid import UUID

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[UUID]], *, k: int = RRF_K
) -> list[tuple[UUID, float]]:
    """Fuse ranked id lists. An id repeated in one list counts at its first rank."""
    scores: dict[UUID, float] = {}
    for ranking in rankings:
        seen: set[UUID] = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    # Equal scores keep first-seen order. dict preserves that, and sorted is stable.
    return sorted(scores.items(), key=lambda item: -item[1])
