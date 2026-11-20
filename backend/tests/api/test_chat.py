import json
from datetime import UTC, datetime
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import CurrentUser, get_current_user
from app.chat.messages import (
    IncomingMessage,
    IncomingPart,
    assistant_parts,
    latest_user_text,
)
from app.chat.orchestrator import CompletedTurn
from app.chat.streaming import text_pieces
from app.database.chats import (
    ChatStoreError,
    CitationInsert,
    StoredMessage,
    ThreadForbidden,
    ThreadNotFound,
    ThreadRecord,
)
from app.grounding.validator import GroundingError
from app.main import app

USER_ID = UUID("6b1f3c2a-7e4d-4a1b-9c8d-0e1f2a3b4c5d")
THREAD_ID = UUID("11111111-1111-4111-8111-111111111111")
OTHER_THREAD_ID = UUID("22222222-2222-4222-8222-222222222222")
NOW = datetime(2026, 11, 11, tzinfo=UTC)

USER = CurrentUser(id=USER_ID, email="analyst@example.com", access_token="token")


def _thread() -> ThreadRecord:
    return ThreadRecord(
        id=THREAD_ID,
        title="New chat",
        created_at=NOW,
        updated_at=NOW,
    )


@pytest.fixture
def client():
    app.dependency_overrides[get_current_user] = lambda: USER
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_latest_user_text_requires_a_user_message() -> None:
    with pytest.raises(ValueError, match="user message"):
        latest_user_text([])
    text = latest_user_text(
        [
            IncomingMessage(
                id="a",
                role="user",
                parts=[IncomingPart(type="text", text="What changed?")],
            )
        ]
    )
    assert text == "What changed?"
    pieces = text_pieces("How did revenue change?")
    assert "".join(pieces) == "How did revenue change?"
    parts = assistant_parts("iPhone grew.", [{"ticker": "AAPL", "page": "42"}])
    assert parts[0] == {"type": "text", "text": "iPhone grew."}
    assert parts[1]["type"] == "data-citation"
    assert parts[1]["data"]["ticker"] == "AAPL"
    assert parts[1]["data"]["page"] == "42"


def test_list_and_create_threads(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[str | None] = []

    async def fake_list(_user: CurrentUser) -> list[ThreadRecord]:
        return [_thread()]

    async def fake_create(_user: CurrentUser, title: str | None) -> ThreadRecord:
        created.append(title)
        return _thread()

    monkeypatch.setattr("app.api.chat.list_threads", fake_list)
    monkeypatch.setattr("app.api.chat.create_thread", fake_create)

    listed = client.get("/chat/threads")
    assert listed.status_code == 200
    assert listed.json()["threads"][0]["id"] == str(THREAD_ID)
    assert listed.json()["threads"][0]["title"] == "New chat"

    response = client.post("/chat/threads", json={"title": "Apple 10-K"})
    assert response.status_code == 201
    assert created == ["Apple 10-K"]


def test_message_history(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_require(_user: CurrentUser, thread_id: UUID) -> ThreadRecord:
        assert thread_id == THREAD_ID
        return _thread()

    async def fake_messages(
        _user: CurrentUser, _thread_id: UUID
    ) -> list[StoredMessage]:
        return [
            StoredMessage(
                id=UUID("33333333-3333-4333-8333-333333333333"),
                role="user",
                parts=[{"type": "text", "text": "Hello"}],
            )
        ]

    monkeypatch.setattr("app.api.chat.require_owned_thread", fake_require)
    monkeypatch.setattr("app.api.chat.list_messages", fake_messages)

    response = client.get(f"/chat/threads/{THREAD_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["parts"][0]["text"] == "Hello"


def test_other_users_thread_is_forbidden(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_require(_user: CurrentUser, _thread_id: UUID) -> ThreadRecord:
        raise ThreadForbidden

    monkeypatch.setattr("app.api.chat.require_owned_thread", fake_require)

    response = client.get(f"/chat/threads/{OTHER_THREAD_ID}")

    assert response.status_code == 403
    assert response.json()["detail"] == "You do not have access to this thread"


def test_missing_thread_is_not_found(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_require(_user: CurrentUser, _thread_id: UUID) -> ThreadRecord:
        raise ThreadNotFound

    monkeypatch.setattr("app.api.chat.require_owned_thread", fake_require)

    response = client.get(f"/chat/threads/{THREAD_ID}")

    assert response.status_code == 404


def test_stream_sends_the_answer_and_persists_citations(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved: list[tuple[str, str, list[CitationInsert]]] = []
    chunk_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")

    async def fake_require(_user: CurrentUser, thread_id: UUID) -> ThreadRecord:
        assert thread_id == THREAD_ID
        return _thread()

    async def fake_run(
        _user: CurrentUser,
        _thread: ThreadRecord,
        question: str,
        report=None,
    ) -> CompletedTurn:
        assert question == "How did revenue change?"
        if report is not None:
            await report("retrieve", "Retrieved 1 passage")
        data = {
            "chunkId": str(chunk_id),
            "ticker": "AAPL",
            "page": "42",
            "excerpt": "iPhone",
        }
        return CompletedTurn(
            answer="iPhone revenue grew.",
            parts=assistant_parts("iPhone revenue grew.", [data]),
            citations=(CitationInsert(chunk_id=chunk_id, excerpt="iPhone"),),
        )

    async def fake_append(
        _user: CurrentUser,
        _thread: ThreadRecord,
        user_text: str,
        assistant_text: str,
        *,
        parts: list[dict[str, object]],
        citations: list[CitationInsert],
    ) -> None:
        assert parts[1]["type"] == "data-citation"
        assert any(
            part.get("type") == "data-status"
            and isinstance(part.get("data"), dict)
            and part["data"].get("phase") == "stream"
            for part in parts
        )
        saved.append((user_text, assistant_text, citations))

    monkeypatch.setattr("app.api.chat.require_owned_thread", fake_require)
    monkeypatch.setattr("app.api.chat.run_turn", fake_run)
    monkeypatch.setattr("app.api.chat.append_turn", fake_append)
    monkeypatch.setattr("app.chat.streaming.asyncio.sleep", _no_sleep)

    response = client.post(
        "/chat/stream",
        json={
            "threadId": str(THREAD_ID),
            "messages": [
                {
                    "id": "user-1",
                    "role": "user",
                    "parts": [{"type": "text", "text": "How did revenue change?"}],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    deltas = [event["delta"] for event in events if event.get("type") == "text-delta"]
    assert "".join(deltas) == "iPhone revenue grew."
    citations = [event for event in events if event.get("type") == "data-citation"]
    assert citations[0]["data"]["ticker"] == "AAPL"
    assert citations[0]["data"]["page"] == "42"
    statuses = [event for event in events if event.get("type") == "data-status"]
    assert [event["data"]["detail"] for event in statuses] == [
        "Retrieved 1 passage",
        "Writing the answer",
        "Saving the answer and citations",
    ]
    assert events[0]["type"] == "start"
    assert "data: [DONE]" in response.text
    assert saved == [
        (
            "How did revenue change?",
            "iPhone revenue grew.",
            [CitationInsert(chunk_id=chunk_id, excerpt="iPhone")],
        )
    ]


def test_stream_does_not_persist_when_grounding_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_require(_user: CurrentUser, _thread_id: UUID) -> ThreadRecord:
        return _thread()

    async def fake_run(*_args: object, **_kwargs: object) -> CompletedTurn:
        raise GroundingError("Chunk was not retrieved")

    async def fake_append(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("should not persist")

    monkeypatch.setattr("app.api.chat.require_owned_thread", fake_require)
    monkeypatch.setattr("app.api.chat.run_turn", fake_run)
    monkeypatch.setattr("app.api.chat.append_turn", fake_append)

    response = client.post(
        "/chat/stream",
        json={
            "threadId": str(THREAD_ID),
            "messages": [
                {
                    "id": "user-1",
                    "role": "user",
                    "parts": [{"type": "text", "text": "Hi"}],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "could not be grounded" in response.text
    assert "data: [DONE]" in response.text


def test_stream_rejects_a_missing_user_message(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_require(_user: CurrentUser, _thread_id: UUID) -> ThreadRecord:
        return _thread()

    monkeypatch.setattr("app.api.chat.require_owned_thread", fake_require)

    response = client.post(
        "/chat/stream",
        json={"threadId": str(THREAD_ID), "messages": []},
    )

    assert response.status_code == 422


def test_stream_forbidden_does_not_persist(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_require(_user: CurrentUser, _thread_id: UUID) -> ThreadRecord:
        raise ThreadForbidden

    async def fake_append(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("should not persist")

    monkeypatch.setattr("app.api.chat.require_owned_thread", fake_require)
    monkeypatch.setattr("app.api.chat.append_turn", fake_append)

    response = client.post(
        "/chat/stream",
        json={
            "threadId": str(OTHER_THREAD_ID),
            "messages": [
                {
                    "id": "user-1",
                    "role": "user",
                    "parts": [{"type": "text", "text": "Hi"}],
                }
            ],
        },
    )

    assert response.status_code == 403


def test_unauthenticated_thread_list_is_rejected() -> None:
    with TestClient(app) as test_client:
        response = test_client.get("/chat/threads")
    assert response.status_code == 401


def test_store_failure_is_a_bad_gateway(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_list(_user: CurrentUser) -> list[ThreadRecord]:
        raise ChatStoreError("down")

    monkeypatch.setattr("app.api.chat.list_threads", fake_list)

    response = client.get("/chat/threads")

    assert response.status_code == 502


async def _no_sleep(_seconds: float) -> None:
    return None
