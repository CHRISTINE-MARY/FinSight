"""Chunk SEC HTML through Docling's hierarchical and hybrid chunkers.

HierarchicalChunker walks the parsed document. HybridChunker then splits
chunks that exceed the embedding token limit and merges short neighbors that
share the same headings. The text stored for each chunk is the contextualized
form, which repeats that heading path.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import tiktoken
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, HTMLFormatOption
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    HierarchicalChunker,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc.document import DoclingDocument

from app.ingestion.types import PreparedChunk

# text-embedding-3-small accepts 8191 tokens. Retrieval chunks stay much smaller.
MAX_CHUNK_TOKENS = 512


class _MarkdownTables(ChunkingSerializerProvider):
    """Serialize SEC tables as Markdown instead of row/column triplets."""

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


def openai_tokenizer(max_tokens: int = MAX_CHUNK_TOKENS) -> OpenAITokenizer:
    """Tokenizer aligned with text-embedding-3-small (cl100k_base)."""
    return OpenAITokenizer(
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        max_tokens=max_tokens,
    )


def hierarchical_chunks(document: DoclingDocument) -> list[PreparedChunk]:
    """One chunk per document element, with list items merged."""
    chunker = HierarchicalChunker(serializer_provider=_MarkdownTables())
    return list(_prepare(chunker, document))


def hybrid_chunks(
    document: DoclingDocument,
    *,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[PreparedChunk]:
    """Token-limited chunks refined from the hierarchical chunker."""
    chunker = HybridChunker(
        tokenizer=openai_tokenizer(max_tokens),
        merge_peers=True,
        repeat_table_header=True,
        serializer_provider=_MarkdownTables(),
    )
    return list(_prepare(chunker, document))


def document_from_html(path: Path) -> DoclingDocument:
    result = _html_converter().convert(path)
    if result.status.name != "SUCCESS":
        raise RuntimeError(f"{path.name} converted with status {result.status.name}")
    return result.document


def document_from_html_string(html: str, *, name: str = "fixture.html") -> DoclingDocument:
    result = _html_converter().convert_string(html, format=InputFormat.HTML, name=name)
    if result.status.name != "SUCCESS":
        raise RuntimeError(f"{name} converted with status {result.status.name}")
    return result.document


def _prepare(chunker: HierarchicalChunker | HybridChunker, document: DoclingDocument):
    index = 0
    for chunk in chunker.chunk(document):
        text = chunker.contextualize(chunk).strip()
        if not text:
            continue
        yield PreparedChunk(
            index=index,
            text=text,
            token_count=chunker.tokenizer.count_tokens(text)
            if isinstance(chunker, HybridChunker)
            else openai_tokenizer().count_tokens(text),
            headings=list(chunk.meta.headings or []),
            pages=_pages(chunk),
            doc_item_refs=[item.self_ref for item in chunk.meta.doc_items],
        )
        index += 1


def _pages(chunk: object) -> list[int]:
    pages: set[int] = set()
    meta = getattr(chunk, "meta", None)
    for item in getattr(meta, "doc_items", []) or []:
        for prov in getattr(item, "prov", []) or []:
            page_no = getattr(prov, "page_no", None)
            if isinstance(page_no, int):
                pages.add(page_no)
    return sorted(pages)


@lru_cache(maxsize=1)
def _html_converter() -> DocumentConverter:
    return DocumentConverter(
        allowed_formats=[InputFormat.HTML],
        format_options={InputFormat.HTML: HTMLFormatOption()},
    )
