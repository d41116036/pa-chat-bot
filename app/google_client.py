"""Compatibility wrappers that call gemini-app over HTTP."""

from typing import Any, List, Optional, Tuple

from app.gemini_service import (
    GoogleAIConfigurationError,
    GoogleAIServiceError,
    generate_chat_reply as _remote_generate_chat_reply,
    generate_chat_reply_async as _remote_generate_chat_reply_async,
    summarize_text as _remote_summarize_text,
)

__all__ = [
    "GoogleAIConfigurationError",
    "GoogleAIServiceError",
    "summarize_text",
    "generate_chat_reply",
    "generate_chat_reply_async",
]


def _build_chat_prompt(
    message: str,
    history: Optional[List[Any]] = None,
    context: Optional[str] = None,
) -> str:
    history_lines = []
    for turn in history or []:
        role = str(turn.get("role", "user")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        speaker = "User" if role == "user" else "Assistant"
        history_lines.append("{}: {}".format(speaker, content))

    history_block = (
        "\n".join(history_lines) if history_lines else "No previous context."
    )
    context_block = (
        context.strip() if context and context.strip() else "No retrieved documents."
    )
    return (
        "You are a helpful assistant. Use previous conversation context when relevant. "
        "Prefer the retrieved documents when answering. If the documents do not contain "
        "the answer, say you do not know based on the available notes.\n\n"
        "Retrieved documents:\n{}\n\n"
        "Conversation history:\n{}\n\n"
        "User message:\n{}"
    ).format(context_block, history_block, message)


def summarize_text(text: str, max_words: Optional[int] = 150) -> Tuple[str, str]:
    return _remote_summarize_text(text, max_words)


def generate_chat_reply(
    message: str,
    history: Optional[list] = None,
    context: Optional[str] = None,
) -> Tuple[str, str]:
    return _remote_generate_chat_reply(_build_chat_prompt(message, history, context))


async def generate_chat_reply_async(
    message: str,
    history: Optional[list] = None,
    context: Optional[str] = None,
) -> Tuple[str, str]:
    return await _remote_generate_chat_reply_async(
        _build_chat_prompt(message, history, context)
    )
