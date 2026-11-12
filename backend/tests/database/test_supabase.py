import asyncio

import pytest
from supabase.lib.client_options import AsyncClientOptions

from app.config import settings
from app.database.supabase import create_user_client, get_service_client


def test_user_client_uses_anon_key_and_user_jwt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_create(
        url: str,
        key: str,
        options: AsyncClientOptions | None = None,
    ) -> object:
        assert options is not None
        captured["url"] = url
        captured["key"] = key
        captured["authorization"] = options.headers["Authorization"]
        return object()

    monkeypatch.setattr("app.database.supabase.acreate_client", fake_create)

    asyncio.run(create_user_client("  user-jwt  "))

    assert captured["url"] == settings.supabase_url
    assert captured["key"] == settings.supabase_anon_key
    assert captured["authorization"] == "Bearer user-jwt"


def test_user_client_rejects_blank_token() -> None:
    with pytest.raises(ValueError, match="access_token is required"):
        asyncio.run(create_user_client("   "))


def test_service_client_uses_service_role_key_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_create(
        url: str,
        key: str,
        options: AsyncClientOptions | None = None,
    ) -> object:
        assert options is not None
        calls.append(key)
        assert url == settings.supabase_url
        assert "Authorization" not in options.headers
        return object()

    monkeypatch.setattr("app.database.supabase._service_client", None)
    monkeypatch.setattr("app.database.supabase.acreate_client", fake_create)

    async def load_twice() -> None:
        first = await get_service_client()
        second = await get_service_client()
        assert first is second

    asyncio.run(load_twice())

    assert calls == [settings.supabase_service_role_key]
