"""Run pa-chat-bot RAG once and return DeepEval-ready fields."""

from __future__ import annotations

import asyncio
from typing import List, Tuple

from app.agent import run_tool_agent_for_eval


def _chunks_from_tool_results(tool_results: list) -> List[str]:
    chunks: List[str] = []
    for item in tool_results:
        if item.get("name") != "retrieve_documents":
            continue
        result = item.get("result")
        # MCP structured_content is often {"result": ["chunk", ...]}
        if isinstance(result, dict) and "result" in result:
            result = result.get("result")
        if isinstance(result, list):
            for text in result:
                if isinstance(text, dict) and "result" in text:
                    text = text.get("result")
                if isinstance(text, list):
                    chunks.extend(str(part) for part in text if str(part).strip())
                elif text is not None and str(text).strip():
                    chunks.append(str(text).strip())
        elif isinstance(result, str) and result.strip():
            chunks.append(result.strip())
    return chunks


def run_rag_for_eval(question: str) -> Tuple[str, List[str]]:
    """Run the live agent once; return (answer, retrieval_context)."""
    if not question or not str(question).strip():
        raise ValueError("question must not be blank")
    reply, _model, tool_results = asyncio.run(run_tool_agent_for_eval(question))
    retrieval_context = _chunks_from_tool_results(tool_results)
    if not isinstance(retrieval_context, list):
        raise TypeError("retrieval_context must be a list")
    return reply, retrieval_context
