from typing import Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: str
    filename: str


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20_000)


class ChatRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=2_000)
    history: list[ChatTurn] = []
