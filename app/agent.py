"""Async prompt-based agent loop: Gemini chooses tools via MCP."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.gemini_service import generate_chat_reply_async
from app.tool_registry import (
    ToolExecutionError,
    execute_tool,
    list_tool_definitions,
)

logger = logging.getLogger(__name__)

MAX_AGENT_STEPS = 5


class AgentError(Exception):
    """Raised when the agent cannot produce a valid final reply."""


def _format_history(history: Optional[List[Dict[str, Any]]]) -> str:
    lines: List[str] = []
    for turn in history or []:
        role = str(turn.get("role", "user")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        speaker = "User" if role == "user" else "Assistant"
        lines.append("{}: {}".format(speaker, content))
    return "\n".join(lines) if lines else "No previous context."


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def _build_decision_prompt(
    message: str,
    history: Optional[List[Dict[str, Any]]],
    tools: List[Dict[str, Any]],
    tool_results: List[Dict[str, Any]],
    *,
    force_final: bool = False,
) -> str:
    tools_json = json.dumps(tools, indent=2)
    results_json = json.dumps(tool_results, indent=2)
    history_block = _format_history(history)
    tool_names = [str(t.get("name")) for t in tools]

    if force_final:
        instruction = (
            "You MUST respond with a final answer now using the tool results below. "
            'Return ONLY JSON: {"action":"final","reply":"<your answer>"}'
        )
    else:
        instruction = (
            "Decide the next step. Return ONLY one JSON object, no other text.\n"
            "If you need data, call exactly one available tool:\n"
            '  {"action":"tool","name":"<tool_name>","args":{...}}\n'
            "When you can answer the user, return:\n"
            '  {"action":"final","reply":"<your answer>"}\n'
            "Available tool names: {}\n"
            "Typical flow for factual questions: list_namespaces first, then "
            "retrieve_documents with the best namespace, then final.\n"
            "Prefer retrieved documents when answering. If documents do not contain "
            "the answer, say you do not know based on the available notes."
        ).format(", ".join(tool_names) if tool_names else "(none)")

    return (
        "You are a helpful assistant with tools.\n\n"
        "{}\n\n"
        "Available tools:\n{}\n\n"
        "Tool results so far this turn:\n{}\n\n"
        "Conversation history:\n{}\n\n"
        "User message:\n{}"
    ).format(instruction, tools_json, results_json, history_block, message)


async def _ask_for_action(
    message: str,
    history: Optional[List[Dict[str, Any]]],
    tools: List[Dict[str, Any]],
    tool_results: List[Dict[str, Any]],
    *,
    force_final: bool = False,
) -> Tuple[Dict[str, Any], str]:
    prompt = _build_decision_prompt(
        message, history, tools, tool_results, force_final=force_final
    )
    raw, model = await generate_chat_reply_async(prompt)
    try:
        action = _extract_json(raw)
        return action, model
    except (json.JSONDecodeError, ValueError) as first_exc:
        logger.warning("Agent JSON parse failed; retrying once: %s", first_exc)
        repair_prompt = (
            "Your previous reply was not valid JSON.\n"
            "Previous reply:\n{}\n\n"
            "Return ONLY a valid JSON object for the next action "
            "(tool or final), no markdown."
        ).format(raw)
        raw2, model = await generate_chat_reply_async(repair_prompt)
        try:
            action = _extract_json(raw2)
            return action, model
        except (json.JSONDecodeError, ValueError) as second_exc:
            raise AgentError(
                "Gemini did not return valid tool/final JSON: {}".format(second_exc)
            ) from second_exc


async def run_tool_agent(
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    """Run the async tool-selection loop and return (reply, model)."""
    tools = await list_tool_definitions()
    if not tools:
        raise AgentError("arun-mcp-server returned no tools.")

    tool_results: List[Dict[str, Any]] = []
    model = ""

    for step in range(MAX_AGENT_STEPS):
        force_final = step == MAX_AGENT_STEPS - 1
        action, model = await _ask_for_action(
            message, history, tools, tool_results, force_final=force_final
        )
        action_type = str(action.get("action", "")).strip().lower()
        logger.info(
            "Agent step=%s action=%s payload=%s",
            step + 1,
            action_type,
            action,
        )

        if action_type == "final":
            reply = str(action.get("reply", "")).strip()
            if not reply:
                raise AgentError("Gemini returned a final action with an empty reply.")
            return reply, model

        if action_type == "tool":
            if force_final:
                raise AgentError("Max agent steps reached without a final reply.")
            name = str(action.get("name", "")).strip()
            args = action.get("args") or {}
            if not isinstance(args, dict):
                raise AgentError("Tool args must be a JSON object.")
            try:
                result = await execute_tool(name, args)
            except ToolExecutionError as exc:
                result = {"error": str(exc)}
            tool_results.append({"name": name, "args": args, "result": result})
            continue

        raise AgentError(
            "Gemini returned unknown action {!r}. Expected 'tool' or 'final'.".format(
                action_type
            )
        )

    raise AgentError("Agent loop ended without a final reply.")
