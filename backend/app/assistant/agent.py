"""Filing assistant tools. They can only read chunks, not write SQL."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

from pydantic_ai import Agent, PromptedOutput, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider

from app.assistant.deps import DocumentAgentDeps
from app.assistant.outputs import ModelAnswer
from app.chat.status import threadsafe_report
from app.config import NVIDIA_BASE_URL, settings
from app.retrieval.retriever import RetrievedPassage
from app.retrieval.scope import filing_header

_EXCERPT_CHARS = 500
_agent: Agent[DocumentAgentDeps, ModelAnswer] | None = None


def get_agent() -> Agent[DocumentAgentDeps, ModelAnswer]:
    global _agent
    if _agent is None:
        model = OpenAIChatModel(
            settings.nvidia_chat_model,
            provider=OpenAICompatibleProvider(
                "nvidia",
                base_url=NVIDIA_BASE_URL,
                api_key=settings.nvidia_api_key,
            ),
        )
        instructions = (Path(__file__).parent / "instructions.md").read_text(
            encoding="utf-8"
        )
        agent: Agent[DocumentAgentDeps, ModelAnswer] = Agent(
            model,
            deps_type=DocumentAgentDeps,
            output_type=PromptedOutput(ModelAnswer),
            instructions=instructions,
            name="document_copilot",
            # Lightning reasons unless told not to. This assistant should answer
            # from the passages, not spend the call on a hidden chain of thought.
            model_settings={
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
            },
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
    await _note(ctx, "tool", f"search_filings: {cleaned}")
    passages = await _search(ctx, cleaned)
    noun = "passage" if len(passages) == 1 else "passages"
    await _note(ctx, "tool", f"search_filings returned {len(passages)} {noun}")
    return [_summary(ctx, passage) for passage in passages]


async def read_chunk(
    ctx: RunContext[DocumentAgentDeps], chunk: int
) -> dict[str, object]:
    """Read one retrieved chunk in full.

    Args:
        chunk: Passage number from this request.
    """
    parsed = ctx.deps.numbers.get(chunk)
    if parsed is None:
        return {"error": "Unknown passage number"}
    await _note(ctx, "tool", f"read_chunk {chunk}")
    passage = await _read(ctx, parsed)
    if passage is None:
        return {"error": "Chunk was not found"}
    return _summary(ctx, passage, full=True)


async def read_surrounding_chunks(
    ctx: RunContext[DocumentAgentDeps], chunk: int
) -> list[dict[str, object]] | dict[str, str]:
    """Read the chunks immediately before and after a chunk.

    Args:
        chunk: Passage number whose neighbors should be read.
    """
    parsed = ctx.deps.numbers.get(chunk)
    if parsed is None:
        return {"error": "Unknown passage number"}
    await _note(ctx, "tool", f"read_surrounding_chunks {chunk}")
    passages = await _surrounding(ctx, parsed)
    return [_summary(ctx, passage, full=True) for passage in passages]


def _remember(ctx: RunContext[DocumentAgentDeps], passage: RetrievedPassage) -> None:
    ctx.deps.retrieved[passage.chunk_id] = passage


async def _note(ctx: RunContext[DocumentAgentDeps], phase: str, detail: str) -> None:
    if ctx.deps.report is not None:
        await ctx.deps.report(phase, detail)


async def _search(
    ctx: RunContext[DocumentAgentDeps], query: str
) -> list[RetrievedPassage]:
    progress = threadsafe_report(ctx.deps.report) if ctx.deps.report else None
    passages = await asyncio.to_thread(
        ctx.deps.retriever.search, query, progress=progress
    )
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


def _summary(
    ctx: RunContext[DocumentAgentDeps],
    passage: RetrievedPassage,
    *,
    full: bool = False,
) -> dict[str, object]:
    content = passage.content if full else passage.content[:_EXCERPT_CHARS]
    header = filing_header(
        passage.company_name, passage.filing_type, passage.fiscal_year
    )
    return {
        "chunk": ctx.deps.assign(passage.chunk_id),
        "ticker": passage.ticker,
        "company_name": passage.company_name,
        "filing_type": passage.filing_type,
        "filing_date": passage.filing_date.isoformat(),
        "fiscal_year": passage.fiscal_year,
        "accession_number": passage.accession_number,
        "page": _text(passage.metadata.get("page")),
        "section": _text(passage.metadata.get("section")),
        "content": f"{header}\n{content}",
    }


def _text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        joined = ", ".join(parts)
        return joined or None
    text = str(value).strip()
    return text or None
