"""Dependencies passed into the filing assistant for one turn."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from uuid import UUID

from app.retrieval.retriever import DocumentRetriever, RetrievedPassage

StatusReport = Callable[[str, str], Awaitable[None]]


@dataclass
class DocumentAgentDeps:
    user_id: str
    thread_id: str
    retriever: DocumentRetriever
    retrieved: dict[UUID, RetrievedPassage] = field(default_factory=dict)
    report: StatusReport | None = None
