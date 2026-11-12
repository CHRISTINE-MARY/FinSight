from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase_auth.errors import AuthApiError

from app.database.supabase import create_user_client

_bearer = HTTPBearer()


@dataclass(frozen=True, slots=True)
class CurrentUser:
    id: UUID
    email: str
    access_token: str


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
) -> CurrentUser:
    """Verify `Authorization: Bearer` with Supabase Auth and return that user."""
    access_token = credentials.credentials.strip()
    try:
        client = await create_user_client(access_token)
        response = await client.auth.get_user(access_token)
    except ValueError as exc:
        raise _unauthorized("Invalid authentication credentials") from exc
    except AuthApiError as exc:
        if exc.status >= 500:
            raise
        raise _unauthorized("Invalid authentication credentials") from exc

    email = (response.user.email or "").strip() if response is not None else ""
    if response is None or not email:
        raise _unauthorized("Invalid authentication credentials")
    try:
        user_id = UUID(response.user.id)
    except ValueError as exc:
        raise _unauthorized("Invalid authentication credentials") from exc
    return CurrentUser(id=user_id, email=email, access_token=access_token)
