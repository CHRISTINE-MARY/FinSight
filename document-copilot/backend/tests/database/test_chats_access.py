import asyncio
from datetime import UTC, datetime
from uuid import UUID

import pytest

from app.auth.dependencies import CurrentUser
from app.database import chats
from app.database.chats import (
    ThreadForbidden,
    ThreadNotFound,
    list_threads,
    require_owned_thread,
)

USER_ID = UUID("6b1f3c2a-7e4d-4a1b-9c8d-0e1f2a3b4c5d")
OTHER_ID = UUID("99999999-9999-4999-8999-999999999999")
THREAD_ID = UUID("11111111-1111-4111-8111-111111111111")
NOW = datetime(2026, 11, 11, tzinfo=UTC).isoformat()
USER = CurrentUser(id=USER_ID, email="analyst@example.com", access_token="token")


class _Query:
    def __init__(self, result: object) -> None:
        self._result = result

    def select(self, *_args: object, **_kwargs: object) -> "_Query":
        return self

    def eq(self, *_args: object, **_kwargs: object) -> "_Query":
        return self

    def maybe_single(self) -> "_Query":
        return self

    async def execute(self) -> object:
        return self._result


class _Client:
    def __init__(self, result: object) -> None:
        self._result = result

    def table(self, _name: str) -> _Query:
        return _Query(self._result)


class _Row:
    def __init__(self, data: dict[str, object] | None) -> None:
        self.data = data


def _patch(monkeypatch: pytest.MonkeyPatch, visible: object, owner: object) -> None:
    async def user_client(_token: str) -> _Client:
        return _Client(visible)

    async def service_client() -> _Client:
        return _Client(owner)

    monkeypatch.setattr(chats, "create_user_client", user_client)
    monkeypatch.setattr(chats, "get_service_client", service_client)


def test_list_threads_skips_chats_with_no_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    empty_id = UUID("44444444-4444-4444-8444-444444444444")

    class _Rows:
        def __init__(self, data: list[dict[str, object]]) -> None:
            self.data = data

    class _ListQuery:
        def __init__(self, table_name: str) -> None:
            self.table_name = table_name

        def select(self, *_args: object, **_kwargs: object) -> "_ListQuery":
            return self

        def eq(self, *_args: object, **_kwargs: object) -> "_ListQuery":
            return self

        def order(self, *_args: object, **_kwargs: object) -> "_ListQuery":
            return self

        async def execute(self) -> _Rows:
            if self.table_name == "chat_messages":
                return _Rows([{"thread_id": str(THREAD_ID)}])
            return _Rows(
                [
                    {
                        "id": str(THREAD_ID),
                        "title": "How did Apple revenue change?",
                        "created_at": NOW,
                        "updated_at": NOW,
                    },
                    {
                        "id": str(empty_id),
                        "title": "New chat",
                        "created_at": NOW,
                        "updated_at": NOW,
                    },
                ]
            )

    class _ListClient:
        def table(self, name: str) -> _ListQuery:
            return _ListQuery(name)

    async def user_client(_token: str) -> _ListClient:
        return _ListClient()

    monkeypatch.setattr(chats, "create_user_client", user_client)

    threads = asyncio.run(list_threads(USER))

    assert [thread.id for thread in threads] == [THREAD_ID]


def test_owned_thread_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(
        monkeypatch,
        _Row(
            {
                "id": str(THREAD_ID),
                "title": "New chat",
                "created_at": NOW,
                "updated_at": NOW,
            }
        ),
        None,
    )

    thread = asyncio.run(require_owned_thread(USER, THREAD_ID))

    assert thread.id == THREAD_ID


def test_another_users_thread_is_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, None, _Row({"user_id": str(OTHER_ID)}))

    with pytest.raises(ThreadForbidden):
        asyncio.run(require_owned_thread(USER, THREAD_ID))


def test_unknown_thread_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, None, None)

    with pytest.raises(ThreadNotFound):
        asyncio.run(require_owned_thread(USER, THREAD_ID))
