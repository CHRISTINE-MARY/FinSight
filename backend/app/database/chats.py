from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from postgrest.exceptions import APIError

from app.auth.dependencies import CurrentUser
from app.database.supabase import create_user_client, get_service_client

DEFAULT_THREAD_TITLE = "New chat"


class ThreadNotFound(Exception):
    pass


class ThreadForbidden(Exception):
    pass


class ChatStoreError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ThreadRecord:
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: UUID
    role: str
    parts: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class CitationInsert:
    chunk_id: UUID
    excerpt: str


def _store_error(exc: APIError) -> ChatStoreError:
    message = getattr(exc, "message", None)
    return ChatStoreError(message if isinstance(message, str) else str(exc))


def _as_uuid(value: object) -> UUID:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    raise ChatStoreError("Unexpected id from the database")


def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ChatStoreError("Unexpected timestamp from the database")


def _row(response: object) -> dict[str, object] | None:
    if response is None:
        return None
    data = getattr(response, "data", None)
    if data is None:
        return None
    if isinstance(data, dict):
        return data
    raise ChatStoreError("Unexpected row from the database")


def _rows(response: object) -> list[dict[str, object]]:
    data = getattr(response, "data", None)
    if not data:
        return []
    if isinstance(data, list):
        return data
    raise ChatStoreError("Unexpected rows from the database")


def _thread(row: dict[str, object]) -> ThreadRecord:
    return ThreadRecord(
        id=_as_uuid(row["id"]),
        title=str(row["title"]),
        created_at=_as_datetime(row["created_at"]),
        updated_at=_as_datetime(row["updated_at"]),
    )


async def list_threads(user: CurrentUser) -> list[ThreadRecord]:
    client = await create_user_client(user.access_token)
    try:
        response = await (
            client.table("chat_threads")
            .select("id,title,created_at,updated_at")
            .eq("user_id", str(user.id))
            .order("updated_at", desc=True)
            .execute()
        )
        started = await client.table("chat_messages").select("thread_id").execute()
    except APIError as exc:
        raise _store_error(exc) from exc
    started_ids = {
        str(row["thread_id"])
        for row in _rows(started)
        if row.get("thread_id") is not None
    }
    return [_thread(row) for row in _rows(response) if str(row["id"]) in started_ids]


async def create_thread(user: CurrentUser, title: str | None) -> ThreadRecord:
    cleaned = (title or "").strip() or DEFAULT_THREAD_TITLE
    client = await create_user_client(user.access_token)
    try:
        response = await (
            client.table("chat_threads")
            .insert(
                {
                    "user_id": str(user.id),
                    "title": cleaned,
                }
            )
            .execute()
        )
    except APIError as exc:
        raise _store_error(exc) from exc
    rows = _rows(response)
    if not rows:
        raise ChatStoreError("Creating a thread did not return a row")
    return _thread(rows[0])


async def require_owned_thread(user: CurrentUser, thread_id: UUID) -> ThreadRecord:
    """Return the thread, or raise when it is missing or owned by someone else."""
    client = await create_user_client(user.access_token)
    try:
        visible = await (
            client.table("chat_threads")
            .select("id,title,created_at,updated_at")
            .eq("id", str(thread_id))
            .maybe_single()
            .execute()
        )
    except APIError as exc:
        raise _store_error(exc) from exc
    row = _row(visible)
    if row is not None:
        return _thread(row)

    service = await get_service_client()
    try:
        owner = await (
            service.table("chat_threads")
            .select("user_id")
            .eq("id", str(thread_id))
            .maybe_single()
            .execute()
        )
    except APIError as exc:
        raise _store_error(exc) from exc
    owner_row = _row(owner)
    if owner_row is None:
        raise ThreadNotFound
    if str(owner_row["user_id"]) != str(user.id):
        raise ThreadForbidden
    raise ChatStoreError("Thread is not visible to its owner")


async def list_messages(user: CurrentUser, thread_id: UUID) -> list[StoredMessage]:
    client = await create_user_client(user.access_token)
    try:
        response = await (
            client.table("chat_messages")
            .select("id,role,parts")
            .eq("thread_id", str(thread_id))
            .order("position")
            .execute()
        )
    except APIError as exc:
        raise _store_error(exc) from exc
    messages: list[StoredMessage] = []
    for row in _rows(response):
        parts = row["parts"]
        messages.append(
            StoredMessage(
                id=_as_uuid(row["id"]),
                role=str(row["role"]),
                parts=parts if isinstance(parts, list) else [],
            )
        )
    return messages


async def append_turn(
    user: CurrentUser,
    thread: ThreadRecord,
    user_text: str,
    assistant_text: str,
    *,
    parts: list[dict[str, object]],
    citations: list[CitationInsert],
) -> None:
    """Store the user message, assistant reply, and citation rows after a stream finishes."""
    client = await create_user_client(user.access_token)
    try:
        existing = await (
            client.table("chat_messages")
            .select("position")
            .eq("thread_id", str(thread.id))
            .order("position", desc=True)
            .limit(1)
            .execute()
        )
        rows = _rows(existing)
        next_position = int(rows[0]["position"]) + 1 if rows else 0
        inserted = await (
            client.table("chat_messages")
            .insert(
                [
                    _message_row(
                        thread.id,
                        next_position,
                        "user",
                        user_text,
                        [{"type": "text", "text": user_text}],
                    ),
                    _message_row(
                        thread.id,
                        next_position + 1,
                        "assistant",
                        assistant_text,
                        parts,
                    ),
                ]
            )
            .execute()
        )
        if citations:
            assistant_id = _assistant_message_id(inserted)
            await (
                client.table("message_citations")
                .insert(
                    [
                        {
                            "message_id": str(assistant_id),
                            "chunk_id": str(citation.chunk_id),
                            "citation_index": index,
                            "excerpt": citation.excerpt,
                        }
                        for index, citation in enumerate(citations)
                    ]
                )
                .execute()
            )
        update: dict[str, str] = {"updated_at": datetime.now(UTC).isoformat()}
        if next_position == 0 and thread.title == DEFAULT_THREAD_TITLE:
            update["title"] = _title_from(user_text)
        await (
            client.table("chat_threads")
            .update(update)
            .eq("id", str(thread.id))
            .execute()
        )
    except APIError as exc:
        raise _store_error(exc) from exc


def _assistant_message_id(response: object) -> UUID:
    for row in _rows(response):
        if str(row.get("role")) == "assistant":
            return _as_uuid(row["id"])
    raise ChatStoreError("Saving the assistant message did not return its id")


def _message_row(
    thread_id: UUID,
    position: int,
    role: str,
    text: str,
    parts: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "thread_id": str(thread_id),
        "position": position,
        "role": role,
        "content": text,
        "parts": parts,
    }


def _title_from(user_text: str) -> str:
    title = " ".join(user_text.split())
    if len(title) <= 80:
        return title
    return title[:77].rstrip() + "..."
