"""Filing assistant tools. They can only read chunks, not write SQL."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

from pydantic_ai import Agent, PromptedOutput, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from app.assistant.deps import DocumentAgentDeps
from app.assistant.outputs import GroundedAnswer
from app.config import settings
from app.retrieval.retriever import RetrievedPassage

_INSTRUCTIONS = (Path(__file__).parent / "instructions.md").read_text(encoding="utf-8")
_EXCERPT_CHARS = 500
_agent: Agent[DocumentAgentDeps, GroundedAnswer] | None = None


def get_agent() -> Agent[DocumentAgentDeps, GroundedAnswer]:
    global _agent
    if _agent is None:
        model = OpenAIChatModel(
            settings.openrouter_model,
            provider=OpenRouterProvider(api_key=settings.openrouter_api_key),
        )
        agent: Agent[DocumentAgentDeps, GroundedAnswer] = Agent(
            model,
            deps_type=DocumentAgentDeps,
            output_type=PromptedOutput(GroundedAnswer),
            instructions=_INSTRUCTIONS,
            name="document_copilot",
            retries={"output": 2},
        )
        agent.tool(search_filings)
        agent.tool(read_chunk)
        agent.tool(read_surrounding_chunks)
        _agent = agent
    return _agent


async def search_filings(
    ctx: RunContext[DocumentAgentDeps], query: str
) -> list[dict[str, object]]:
    """Search the filing corpus and return the top matching chunks.

    Args:
        query: Analyst question or search terms.
    """
    cleaned = query.strip()
    if not cleaned:
        return []
    passages = await _search(ctx, cleaned)
    return [_summary(passage) for passage in passages]


async def read_chunk(
    ctx: RunContext[DocumentAgentDeps], chunk_id: str
) -> dict[str, object]:
    """Read one retrieved chunk in full.

    Args:
        chunk_id: Chunk id from search_filings or the prompt.
    """
    parsed = _parse_chunk_id(chunk_id)
    if parsed is None:
        return {"error": "Unknown chunk id"}
    passage = await _read(ctx, parsed)
    if passage is None:
        return {"error": "Chunk was not found"}
    return _summary(passage, full=True)


async def read_surrounding_chunks(
    ctx: RunContext[DocumentAgentDeps], chunk_id: str
) -> list[dict[str, object]] | dict[str, str]:
    """Read the chunks immediately before and after a chunk.

    Args:
        chunk_id: Chunk id whose neighbors should be read.
    """
    parsed = _parse_chunk_id(chunk_id)
    if parsed is None:
        return {"error": "Unknown chunk id"}
    passages = await _surrounding(ctx, parsed)
    return [_summary(passage, full=True) for passage in passages]


def _remember(ctx: RunContext[DocumentAgentDeps], passage: RetrievedPassage) -> None:
    ctx.deps.retrieved[passage.chunk_id] = passage


async def _search(
    ctx: RunContext[DocumentAgentDeps], query: str
) -> list[RetrievedPassage]:
    passages = await asyncio.to_thread(ctx.deps.retriever.search, query)
    for passage in passages:
        _remember(ctx, passage)
    return passages


async def _read(
    ctx: RunContext[DocumentAgentDeps], chunk_id: UUID
) -> RetrievedPassage | None:
    passage = await asyncio.to_thread(ctx.deps.retriever.read_chunk, chunk_id)
    if passage is not None:
        _remember(ctx, passage)
    return passage


async def _surrounding(
    ctx: RunContext[DocumentAgentDeps], chunk_id: UUID
) -> list[RetrievedPassage]:
    passages = await asyncio.to_thread(ctx.deps.retriever.read_surrounding, chunk_id)
    for passage in passages:
        _remember(ctx, passage)
    return passages


def _summary(passage: RetrievedPassage, *, full: bool = False) -> dict[str, object]:
    content = passage.content if full else passage.content[:_EXCERPT_CHARS]
    return {
        "chunk_id": str(passage.chunk_id),
        "ticker": passage.ticker,
        "company_name": passage.company_name,
        "filing_type": passage.filing_type,
        "filing_date": passage.filing_date.isoformat(),
        "fiscal_year": passage.fiscal_year,
        "accession_number": passage.accession_number,
        "page": _text(passage.metadata.get("page")),
        "section": _text(passage.metadata.get("section")),
        "content": content,
    }


def _parse_chunk_id(value: str) -> UUID | None:
    try:
        return UUID(value)
    except ValueError:
        return None


def _text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        joined = ", ".join(parts)
        return joined or None
    text = str(value).strip()
    return text or None
