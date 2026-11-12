"""Chunk records shared by chunking and metadata. Kept free of Docling imports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    index: int
    text: str
    token_count: int
    headings: list[str]
    pages: list[int]
    doc_item_refs: list[str]
