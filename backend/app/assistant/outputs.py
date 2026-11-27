"""Typed answer the filing assistant must return."""

from __future__ import annotations

from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class Claim(BaseModel):
    """One factual sentence and the numbered passage it comes from."""

    text: str = Field(
        description=(
            "One factual sentence. Copy figures as printed. "
            "Do not put the passage number in the text."
        )
    )
    chunk: int = Field(
        ge=1,
        description=(
            "Passage number printed on a chunk in this request. "
            "Required for every factual claim. Do not invent a number."
        ),
    )
    label: str = Field(
        default="",
        description=(
            "Row name or short name copied from that passage. Do not rebuild the row."
        ),
    )
    figure: str = Field(
        default="",
        description=(
            "The printed number used from that passage. "
            "Leave empty when the claim has no number. Do not rebuild the row."
        ),
    )


class Citation(BaseModel):
    chunk_id: UUID = Field(description="Id of a chunk returned for this request.")
    label: str = Field(
        default="",
        description=(
            "Row name or short name copied from that chunk. Do not rebuild the row."
        ),
    )
    figure: str = Field(
        default="",
        description=(
            "The printed number used from that chunk. "
            "Leave empty when the claim has no number. Do not rebuild the row."
        ),
    )
    excerpt: str = Field(
        default="",
        description="Leave empty. The server copies the original line.",
    )


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


REFUSAL_SENTENCE = "The corpus does not contain enough evidence."


class ModelAnswer(BaseModel):
    """JSON answer. Set claims, or set answer to the refusal sentence. Both empty is invalid."""

    claims: list[Claim] = Field(
        default_factory=list,
        description=(
            "One entry per factual sentence. Each entry's chunk is a passage "
            "number from this request. Leave this empty only when answer is "
            f"exactly: {REFUSAL_SENTENCE}"
        ),
    )
    answer: str = Field(
        default="",
        description=(
            "Empty string when claims is set. When the passages do not contain "
            f"the facts, set this to exactly: {REFUSAL_SENTENCE} "
            "Do not leave this empty when claims is empty."
        ),
    )
    table: AnswerTable | None = Field(
        default=None,
        description=(
            "Required when the user asks for a table or the question compares "
            "years, segments, or companies. Null when a table does not apply."
        ),
    )

    @model_validator(mode="after")
    def require_claims_or_refusal(self) -> Self:
        # {} matches the defaults, so the schema accepts it. Reject it here so
        # the output retry asks for claims or the refusal sentence.
        if any(claim.text.strip() for claim in self.claims):
            return self
        if self.answer.strip() == REFUSAL_SENTENCE:
            return self
        raise ValueError(
            "Set claims to one entry per factual sentence, or set answer to "
            f"exactly {REFUSAL_SENTENCE!r}. An empty object is not an answer."
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
