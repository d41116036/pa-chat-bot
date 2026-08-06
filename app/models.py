from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SummarizeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=120_000,
        description="Long input text to summarize.",
    )
    max_words: Optional[int] = Field(
        default=150,
        ge=20,
        le=1000,
        description="Target maximum words in the summary.",
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank.")
        return value


class SummarizeResponse(BaseModel):
    summary: str
    model: str
    input_char_count: int


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="User message for the chatbot.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional conversation session identifier.",
    )

    @field_validator("message")
    @classmethod
    def validate_message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank.")
        return value


class ChatResponse(BaseModel):
    reply: str
    model: str
    session_id: str
