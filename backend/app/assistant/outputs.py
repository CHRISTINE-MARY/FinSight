"""Typed answer the filing assistant must return."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: UUID = Field(description="Id of a chunk returned for this request")
    excerpt: str = Field(description="Contiguous text copied from that chunk")


class SourcePassage(BaseModel):
    """Filing passage the server attaches after citation checks."""

    chunk_id: UUID
    ticker: str
    company_name: str
    filing_type: str
    filing_date: str
    fiscal_year: int
    accession_number: str
    page: str | None = None
    section: str | None = None
    content: str


class AnswerTable(BaseModel):
    """Comparison grid the server renders as a Markdown table."""

    headers: list[str] = Field(
        default_factory=list,
        description=(
            "Column titles. The first header is the row label, such as Segment. "
            "The rest are the years or categories being compared."
        ),
    )
    rows: list[list[str]] = Field(
        default_factory=list,
        description=(
            "One list of cells per row, the same length as headers. "
            "Copy the figures from the passages."
        ),
    )


class GroundedAnswer(BaseModel):
    answer: str = Field(
        description=(
            "One or two sentences that answer the question. "
            "Do not describe a table here and do not write 'the table below'."
        )
    )
    table: AnswerTable | None = Field(
        default=None,
        description=(
            "Required when the user asks for a table or the question compares "
            "years, segments, or companies. Null when a table does not apply."
        ),
    )
    citations: list[Citation] = Field(default_factory=list)
    cited_passages: list[SourcePassage] = Field(default_factory=list)


def compose_answer(answer: str, table: AnswerTable | None) -> str:
    """Lead sentence plus a Markdown table when the model returned rows."""
    text = answer.strip()
    rendered = _markdown_table(table)
    if not rendered or _has_markdown_table(text):
        return text
    if not text:
        return rendered
    return f"{text}\n\n{rendered}"


def _markdown_table(table: AnswerTable | None) -> str:
    if table is None:
        return ""
    headers = [_cell(header) for header in table.headers]
    if len(headers) < 2:
        return ""
    body: list[str] = []
    width = len(headers)
    for row in table.rows:
        cells = [_cell(cell) for cell in row]
        if not any(cells):
            continue
        if len(cells) < width:
            cells.extend([""] * (width - len(cells)))
        body.append("| " + " | ".join(cells[:width]) + " |")
    if not body:
        return ""
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    return "\n".join([header, separator, *body])


def _has_markdown_table(text: str) -> bool:
    lines = text.splitlines()
    return any(line.strip().startswith("|") for line in lines) and any(
        "---" in line for line in lines
    )


def _cell(value: str) -> str:
    return " ".join(value.replace("|", "/").split())
