_SMALL_TALK_PHRASES = {
    "hi",
    "hello",
    "hey",
    "hiya",
    "howdy",
    "yo",
    "sup",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "thanks",
    "thank you",
    "thanks a lot",
    "thank you so much",
    "bye",
    "goodbye",
    "see you",
    "ok",
    "okay",
}


def is_small_talk(message: str) -> bool:
    """Return True for short greetings/thanks that should skip Pinecone retrieval."""
    cleaned = " ".join(message.strip().lower().split())
    cleaned = cleaned.strip(" .,!?;:")
    return cleaned in _SMALL_TALK_PHRASES
