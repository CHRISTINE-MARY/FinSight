import asyncio
import json
from collections.abc import AsyncIterator


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


async def answer_events(parts: list[dict[str, object]]) -> AsyncIterator[bytes]:
    """Text and citation events. The caller emits ``start`` before these."""
    text = _answer_text(parts)
    text_id = "text"
    yield encode_sse({"type": "text-start", "id": text_id})
    for piece in text_pieces(text):
        yield encode_sse({"type": "text-delta", "id": text_id, "delta": piece})
        await asyncio.sleep(0.02)
    yield encode_sse({"type": "text-end", "id": text_id})
    for part in parts:
        if part.get("type") == "data-citation":
            yield encode_sse(part)


def _answer_text(parts: list[dict[str, object]]) -> str:
    for part in parts:
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            return part["text"]
    return ""
