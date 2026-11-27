from app.ingestion.chunking import (
    document_from_html_string,
    hierarchical_chunks,
    hybrid_chunks,
    tag_cover_chunks,
)
from app.ingestion.types import PreparedChunk

FIXTURE = (
    """
<html><body>
<h1>Net sales</h1>
<p>iPhone, Mac, iPad, Wearables, and Services make up the revenue mix.</p>
<ul><li>First product</li><li>Second product</li></ul>
<h2>Discussion</h2>
<p>"""
    + ("revenue " * 80)
    + """</p>
</body></html>
"""
)


def test_hierarchical_chunker_keeps_headings_and_merges_list_items() -> None:
    document = document_from_html_string(FIXTURE)
    chunks = hierarchical_chunks(document)

    sales = next(chunk for chunk in chunks if "iPhone" in chunk.text)
    assert sales.headings == ["Net sales"]
    listed = next(chunk for chunk in chunks if "First product" in chunk.text)
    assert "Second product" in listed.text


def test_hybrid_chunker_limits_tokens_and_repeats_headings() -> None:
    document = document_from_html_string(FIXTURE)
    hierarchical = hierarchical_chunks(document)
    chunks = hybrid_chunks(document, max_tokens=30)

    assert len(chunks) > len(hierarchical)
    assert all(chunk.token_count <= 30 for chunk in chunks)
    discussion = next(chunk for chunk in chunks if "Discussion" in chunk.text)
    assert discussion.text.startswith("Net sales\nDiscussion")
    assert discussion.headings == ["Net sales", "Discussion"]


def test_chunks_before_item_1_are_tagged_as_the_cover() -> None:
    cover = PreparedChunk(0, "One Apple Park Way", 4, [], [], [])
    body = PreparedChunk(1, "Item 1. Business", 4, ["Item 1"], [], [])

    tagged = tag_cover_chunks([cover, body])

    assert tagged[0].headings == ["Cover"]
    assert tagged[1].headings == ["Item 1"]


def test_a_document_without_an_item_heading_keeps_its_sections() -> None:
    sales = PreparedChunk(0, "Net sales", 2, ["Net sales"], [], [])

    assert tag_cover_chunks([sales])[0].headings == ["Net sales"]
