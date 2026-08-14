import asyncio
import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from app.agent import AgentError, run_tool_agent
from app.db import DatabaseServiceError, get_chat_history, save_chat_message
from app.google_client import (
    GoogleAIConfigurationError,
    GoogleAIServiceError,
    generate_chat_reply_async,
    summarize_text,
)
from app.models import ChatRequest, ChatResponse, SummarizeRequest, SummarizeResponse
from app.tool_registry import ToolExecutionError
from app.text_utils import is_small_talk

router = APIRouter(prefix="/chatbot")
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
async def chat(payload: ChatRequest) -> ChatResponse:
    session_id = payload.session_id or str(uuid4())
    logger.debug(
        "Received chat request payload: session_id=%s message=%s",
        session_id,
        payload.message,
    )

    try:
        history = await asyncio.to_thread(
            get_chat_history, session_id=session_id, limit=20
        )
        logger.info(
            "Chat loaded history session_id=%s history_turns=%d",
            session_id,
            len(history),
        )
        await asyncio.to_thread(
            save_chat_message,
            session_id=session_id,
            role="user",
            content=payload.message,
        )
        logger.debug("Chat saved user message session_id=%s", session_id)

        if is_small_talk(payload.message):
            logger.info(
                "Small-talk message detected; skipping tool agent: session_id=%s",
                session_id,
            )
            reply, model = await generate_chat_reply_async(
                payload.message, history=history
            )
            await asyncio.to_thread(
                save_chat_message,
                session_id=session_id,
                role="assistant",
                content=reply,
            )
            logger.info(
                "Chat small-talk completed session_id=%s model=%r reply_length=%d",
                session_id,
                model,
                len(reply),
            )
            return ChatResponse(reply=reply, model=model, session_id=session_id)

        logger.info(
            "Starting async MCP tool agent session_id=%s message=%r",
            session_id,
            payload.message,
        )
        reply, model = await run_tool_agent(payload.message, history=history)
        await asyncio.to_thread(
            save_chat_message,
            session_id=session_id,
            role="assistant",
            content=reply,
        )
        logger.info(
            "Chat tool-agent completed session_id=%s model=%r reply_length=%d",
            session_id,
            model,
            len(reply),
        )
    except HTTPException:
        raise
    except DatabaseServiceError as exc:
        logger.exception(
            "Chat failed with database error session_id=%s: %s",
            session_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except GoogleAIConfigurationError as exc:
        logger.exception(
            "Chat failed with Gemini configuration error session_id=%s: %s",
            session_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except (AgentError, ToolExecutionError, GoogleAIServiceError) as exc:
        logger.error(
            "Chat failed with agent/tool/gemini error session_id=%s type=%s detail=%s",
            session_id,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "Chat failed with unexpected error session_id=%s: %s",
            session_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Tool agent chat failed: {}".format(exc),
        ) from exc

    return ChatResponse(reply=reply, model=model, session_id=session_id)
