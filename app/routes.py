import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from app.db import DatabaseServiceError, get_chat_history, save_chat_message
from app.models import ChatRequest, ChatResponse, SummarizeRequest, SummarizeResponse
from app.google_client import (
    GoogleAIConfigurationError,
    GoogleAIServiceError,
    choose_namespace,
    generate_chat_reply,
    summarize_text,
)
from app.pinecone_service import (
    PineconeAppServiceError,
    retrieve_documents,
    retrieve_namespaces,
)
from app.text_utils import is_small_talk

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(payload: SummarizeRequest) -> SummarizeResponse:
    logger.debug(
        "Received summarize request payload: text_length=%s max_words=%s text=%s",
        len(payload.text),
        payload.max_words,
        payload.text,
    )

    try:
        summary, model = summarize_text(payload.text, payload.max_words)
    except GoogleAIConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except GoogleAIServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return SummarizeResponse(
        summary=summary,
        model=model,
        input_char_count=len(payload.text),
    )


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    session_id = payload.session_id or str(uuid4())
    logger.debug(
        "Received chat request payload: session_id=%s message=%s",
        session_id,
        payload.message,
    )

    try:
        history = get_chat_history(session_id=session_id, limit=20)
        save_chat_message(session_id=session_id, role="user", content=payload.message)

        if is_small_talk(payload.message):
            logger.info(
                "Small-talk message detected; skipping Pinecone retrieval: session_id=%s",
                session_id,
            )
            reply, model = generate_chat_reply(payload.message, history=history)
            save_chat_message(session_id=session_id, role="assistant", content=reply)
            return ChatResponse(reply=reply, model=model, session_id=session_id)

        namespaces = retrieve_namespaces()
        if not namespaces:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No Pinecone namespaces found. Upload a PDF first.",
            )

        selected_namespace = choose_namespace(payload.message, namespaces)
        logger.info(
            "Chat namespace selected: session_id=%s namespace=%s options=%s",
            session_id,
            selected_namespace,
            namespaces,
        )

        context_parts = retrieve_documents(
            payload.message,
            selected_namespace,
            top_k=3,
        )
        context = "\n\n".join(context_parts)
        logger.info(
            "Retrieved %s context chunks for session_id=%s namespace=%s",
            len(context_parts),
            session_id,
            selected_namespace,
        )

        reply, model = generate_chat_reply(
            payload.message,
            history=history,
            context=context,
        )
        save_chat_message(session_id=session_id, role="assistant", content=reply)
    except HTTPException:
        raise
    except DatabaseServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except GoogleAIConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except (GoogleAIServiceError, PineconeAppServiceError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Pinecone/chat routing failed: {exc}",
        ) from exc

    return ChatResponse(reply=reply, model=model, session_id=session_id)
