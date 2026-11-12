from typing import Annotated, Literal
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.auth.dependencies import CurrentUser, get_current_user
from app.chat.messages import IncomingMessage, latest_user_text
from app.chat.streaming import encode_sse, stub_events, stub_reply
from app.database.chats import (
    ChatStoreError,
    StoredMessage,
    ThreadForbidden,
    ThreadNotFound,
    ThreadRecord,
    append_turn,
    create_thread,
    list_messages,
    list_threads,
    require_owned_thread,
)

router = APIRouter()
log = structlog.get_logger()

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "x-vercel-ai-ui-message-stream": "v1",
    "x-accel-buffering": "no",
}


class ThreadBody(BaseModel):
    title: str | None = None


class ThreadResponse(BaseModel):
    id: UUID
    title: str
    created_at: str = Field(serialization_alias="createdAt")
    updated_at: str = Field(serialization_alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True)


class MessageResponse(BaseModel):
    id: UUID
    role: Literal["user", "assistant"]
    parts: list[dict[str, object]]


class ThreadDetailResponse(ThreadResponse):
    messages: list[MessageResponse]


class ThreadListResponse(BaseModel):
    threads: list[ThreadResponse]


class StreamRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    thread_id: UUID = Field(alias="threadId")
    messages: list[IncomingMessage]


def _thread_response(thread: ThreadRecord) -> ThreadResponse:
    return ThreadResponse(
        id=thread.id,
        title=thread.title,
        created_at=thread.created_at.isoformat(),
        updated_at=thread.updated_at.isoformat(),
    )


def _message_response(message: StoredMessage) -> MessageResponse:
    role: Literal["user", "assistant"] = (
        "assistant" if message.role == "assistant" else "user"
    )
    return MessageResponse(id=message.id, role=role, parts=message.parts)


def _raise_store(exc: Exception) -> None:
    if isinstance(exc, ThreadNotFound):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Thread not found") from exc
    if isinstance(exc, ThreadForbidden):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "You do not have access to this thread",
        ) from exc
    if isinstance(exc, ChatStoreError):
        log.error("chat store failed", error=str(exc))
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Could not reach the database.",
        ) from exc
    raise exc


@router.get("/chat/threads")
async def get_threads(
    user: Annotated[CurrentUser, Depends(get_current_user)],
) -> ThreadListResponse:
    try:
        threads = await list_threads(user)
    except ChatStoreError as exc:
        _raise_store(exc)
    return ThreadListResponse(threads=[_thread_response(thread) for thread in threads])


@router.post("/chat/threads", status_code=status.HTTP_201_CREATED)
async def post_thread(
    body: ThreadBody,
    user: Annotated[CurrentUser, Depends(get_current_user)],
) -> ThreadResponse:
    try:
        thread = await create_thread(user, body.title)
    except ChatStoreError as exc:
        _raise_store(exc)
    return _thread_response(thread)


@router.get("/chat/threads/{thread_id}")
async def get_thread(
    thread_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
) -> ThreadDetailResponse:
    try:
        thread = await require_owned_thread(user, thread_id)
        messages = await list_messages(user, thread_id)
    except (ThreadNotFound, ThreadForbidden, ChatStoreError) as exc:
        _raise_store(exc)
    response = _thread_response(thread)
    return ThreadDetailResponse(
        id=response.id,
        title=response.title,
        created_at=response.created_at,
        updated_at=response.updated_at,
        messages=[_message_response(message) for message in messages],
    )


@router.post("/chat/stream")
async def stream_chat(
    body: StreamRequest,
    user: Annotated[CurrentUser, Depends(get_current_user)],
) -> StreamingResponse:
    try:
        thread = await require_owned_thread(user, body.thread_id)
        user_text = latest_user_text(body.messages)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except (ThreadNotFound, ThreadForbidden, ChatStoreError) as exc:
        _raise_store(exc)

    reply = stub_reply(user_text)

    async def events():
        async for chunk in stub_events(reply):
            yield chunk
        try:
            await append_turn(user, thread, user_text, reply)
        except ChatStoreError as exc:
            log.error("chat store failed", error=str(exc))
            yield encode_sse(
                {
                    "type": "error",
                    "errorText": "Could not save this conversation.",
                }
            )
            yield encode_sse("[DONE]")
            return
        yield encode_sse({"type": "finish", "finishReason": "stop"})
        yield encode_sse("[DONE]")

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )
