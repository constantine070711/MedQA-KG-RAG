"""Request models for the MedQA v2 API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


RetrievalMode = Literal["naive", "local", "global", "hybrid", "mix"]


class HistoryTurn(BaseModel):
    """A completed conversation turn supplied by the browser."""

    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=16000)
    safety_level: str | None = None
    is_final_refusal: bool = False


class ChatRequest(BaseModel):
    """One chat request with bounded multi-turn context."""

    question: str = Field(min_length=1, max_length=2000)
    mode: RetrievalMode = "mix"
    history: list[HistoryTurn] = Field(default_factory=list, max_length=6)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned
