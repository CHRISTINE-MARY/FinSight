from pydantic import BaseModel, ConfigDict, Field


class IncomingPart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    text: str | None = None


class IncomingMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    role: str
    parts: list[IncomingPart] = Field(default_factory=list)


def message_text(message: IncomingMessage) -> str:
    return "".join(
        part.text for part in message.parts if part.type == "text" and part.text
    )


def latest_user_text(messages: list[IncomingMessage]) -> str:
    if not messages or messages[-1].role != "user":
        raise ValueError("A user message is required")
    text = message_text(messages[-1]).strip()
    if not text:
        raise ValueError("A user message is required")
    return text


def assistant_parts(
    answer: str, citations: list[dict[str, object]]
) -> list[dict[str, object]]:
    """AI SDK parts: the answer text, then one data part per citation."""
    parts: list[dict[str, object]] = [{"type": "text", "text": answer}]
    for index, data in enumerate(citations):
        parts.append({"type": "data-citation", "id": f"citation-{index}", "data": data})
    return parts
