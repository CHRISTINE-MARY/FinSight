import asyncio

from supabase import AsyncClient, acreate_client
from supabase.lib.client_options import AsyncClientOptions

from app.config import settings

_service_client: AsyncClient | None = None
_service_lock = asyncio.Lock()


def _client_options(access_token: str | None = None) -> AsyncClientOptions:
    headers = {}
    if access_token is not None:
        headers["Authorization"] = f"Bearer {access_token}"
    # Server-side clients must not refresh or store a browser session.
    return AsyncClientOptions(
        headers=headers,
        auto_refresh_token=False,
        persist_session=False,
    )


async def create_user_client(access_token: str) -> AsyncClient:
    """Client for one request's user.

    The anon key is the `apikey`. The user's access token is `Authorization`,
    so Postgres row-level security applies.
    """
    token = access_token.strip()
    if not token:
        raise ValueError("access_token is required")
    return await acreate_client(
        settings.supabase_url,
        settings.supabase_anon_key,
        options=_client_options(token),
    )


async def get_service_client() -> AsyncClient:
    """Service-role client for privileged writes. Bypasses row-level security.

    Reused for the process lifetime. The key never changes and the client
    holds no user session.
    """
    global _service_client
    if _service_client is not None:
        return _service_client
    async with _service_lock:
        if _service_client is None:
            _service_client = await acreate_client(
                settings.supabase_url,
                settings.supabase_service_role_key,
                options=_client_options(),
            )
        return _service_client
