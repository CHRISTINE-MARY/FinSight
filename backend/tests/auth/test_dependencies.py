from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from supabase_auth.errors import AuthApiError
from supabase_auth.types import User, UserResponse

from app.auth.dependencies import CurrentUser, get_current_user

USER_ID = UUID("6b1f3c2a-7e4d-4a1b-9c8d-0e1f2a3b4c5d")


def _response(
    email: str | None = "analyst@example.com",
    user_id: str = str(USER_ID),
) -> UserResponse:
    return UserResponse(
        user=User(
            id=user_id,
            app_metadata={},
            user_metadata={},
            aud="authenticated",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            email=email,
        )
    )


class _Auth:
    def __init__(
        self,
        response: UserResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.jwt: str | None = None

    async def get_user(self, jwt: str) -> UserResponse | None:
        self.jwt = jwt
        if self.error is not None:
            raise self.error
        return self.response


class _Client:
    def __init__(self, auth: _Auth) -> None:
        self.auth = auth


def _client() -> TestClient:
    app = FastAPI()

    @app.get("/me")
    async def me(
        user: Annotated[CurrentUser, Depends(get_current_user)],
    ) -> dict[str, str]:
        return {
            "id": str(user.id),
            "email": user.email,
            "access_token": user.access_token,
        }

    return TestClient(app)


def test_missing_bearer_token_is_rejected() -> None:
    response = _client().get("/me")

    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"
    assert response.headers["www-authenticate"] == "Bearer"


def test_non_bearer_scheme_is_rejected() -> None:
    response = _client().get("/me", headers={"Authorization": "Basic abc"})

    assert response.status_code == 401


def test_valid_bearer_token_returns_verified_user(monkeypatch: pytest.MonkeyPatch) -> None:
    auth = _Auth(_response())
    tokens: list[str] = []

    async def fake_create_user_client(access_token: str) -> _Client:
        tokens.append(access_token)
        return _Client(auth)

    monkeypatch.setattr(
        "app.auth.dependencies.create_user_client",
        fake_create_user_client,
    )

    response = _client().get("/me", headers={"Authorization": "Bearer user-jwt"})

    assert response.status_code == 200
    assert response.json() == {
        "id": str(USER_ID),
        "email": "analyst@example.com",
        "access_token": "user-jwt",
    }
    assert tokens == ["user-jwt"]
    assert auth.jwt == "user-jwt"


def test_rejected_token_is_unauthorized(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_create_user_client(access_token: str) -> _Client:
        return _Client(_Auth(error=AuthApiError("invalid", 401, None)))

    monkeypatch.setattr(
        "app.auth.dependencies.create_user_client",
        fake_create_user_client,
    )

    response = _client().get("/me", headers={"Authorization": "Bearer nope"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid authentication credentials"


def test_auth_outage_is_not_hidden_as_unauthorized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_create_user_client(access_token: str) -> _Client:
        return _Client(_Auth(error=AuthApiError("unavailable", 503, None)))

    monkeypatch.setattr(
        "app.auth.dependencies.create_user_client",
        fake_create_user_client,
    )

    with pytest.raises(AuthApiError):
        _client().get("/me", headers={"Authorization": "Bearer user-jwt"})


def test_user_without_email_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_create_user_client(access_token: str) -> _Client:
        return _Client(_Auth(_response(email=None)))

    monkeypatch.setattr(
        "app.auth.dependencies.create_user_client",
        fake_create_user_client,
    )

    response = _client().get("/me", headers={"Authorization": "Bearer user-jwt"})

    assert response.status_code == 401
