import asyncio
import json
from collections.abc import AsyncIterator
from uuid import uuid4


def stub_reply(user_text: str) -> str:
    preview = " ".join(user_text.split())
    if len(preview) > 80:
        preview = preview[:77].rstrip() + "..."
    return (
        f'Stub reply: I received "{preview}". '
        "Document retrieval is not connected yet."
    )


def text_pieces(text: str) -> list[str]:
    words = text.split(" ")
    return [
        word if index == len(words) - 1 else f"{word} "
        for index, word in enumerate(words)
    ]


def encode_sse(event: dict[str, object] | str) -> bytes:
    if isinstance(event, str):
        payload = event
    else:
        payload = json.dumps(event, separators=(",", ":"))
    return f"data: {payload}\n\n".encode()


async def stub_events(text: str) -> AsyncIterator[bytes]:
    """AI SDK UI message stream for a fully known stub reply."""
    message_id = str(uuid4())
    text_id = "text"
    yield encode_sse({"type": "start", "messageId": message_id})
    yield encode_sse({"type": "text-start", "id": text_id})
    for piece in text_pieces(text):
        yield encode_sse({"type": "text-delta", "id": text_id, "delta": piece})
        await asyncio.sleep(0.02)
    yield encode_sse({"type": "text-end", "id": text_id})
