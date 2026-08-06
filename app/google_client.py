"""Compatibility wrappers that call gemini-app over HTTP."""

from app.gemini_service import (
    GoogleAIConfigurationError,
    GoogleAIServiceError,
    generate_chat_reply as _remote_generate_chat_reply,
    summarize_text as _remote_summarize_text,
)

__all__ = [
    "GoogleAIConfigurationError",
    "GoogleAIServiceError",
    "summarize_text",
    "generate_chat_reply",
    "choose_namespace",
]


def summarize_text(text: str, max_words: int | None = 150) -> tuple[str, str]:
    return _remote_summarize_text(text, max_words)


def generate_chat_reply(
    message: str,
    history: list[dict] | None = None,
    context: str | None = None,
) -> tuple[str, str]:
    history_lines: list[str] = []
    for turn in history or []:
        role = str(turn.get("role", "user")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        speaker = "User" if role == "user" else "Assistant"
        history_lines.append(f"{speaker}: {content}")

    history_block = "\n".join(history_lines) if history_lines else "No previous context."
    context_block = (
        context.strip() if context and context.strip() else "No retrieved documents."
    )
    prompt = (
        "You are a helpful assistant. Use previous conversation context when relevant. "
        "Prefer the retrieved documents when answering. If the documents do not contain "
        "the answer, say you do not know based on the available notes.\n\n"
        f"Retrieved documents:\n{context_block}\n\n"
        f"Conversation history:\n{history_block}\n\n"
        f"User message:\n{message}"
    )
    return _remote_generate_chat_reply(prompt)


def choose_namespace(message: str, namespaces: list[str]) -> str:
    if not namespaces:
        raise GoogleAIConfigurationError("No Pinecone namespaces are available.")
    if len(namespaces) == 1:
        return namespaces[0]

    namespace_list = "\n".join(f"- {name}" for name in namespaces)
    prompt = (
        "You route user questions to the best document namespace.\n"
        "Choose exactly one namespace from the list below.\n"
        "Reply with ONLY the namespace name, nothing else.\n\n"
        f"Available namespaces:\n{namespace_list}\n\n"
        f"User message:\n{message}"
    )
    choice, _model = _remote_generate_chat_reply(prompt)
    choice = choice.strip().strip('"').strip("'")

    if choice in namespaces:
        return choice
    for name in namespaces:
        if name in choice:
            return name
    raise GoogleAIServiceError(
        f"Model did not return a valid namespace. Got: {choice!r}"
    )
