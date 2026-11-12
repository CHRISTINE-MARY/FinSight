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
