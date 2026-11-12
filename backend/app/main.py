import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.api.chat import router as chat_router
from app.config import settings

app = FastAPI(title="Document Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-vercel-ai-ui-message-stream"],
)

app.include_router(chat_router)


class Health(BaseModel):
    status: str


@app.get("/health")
async def health() -> Health:
    return Health(status="ok")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
