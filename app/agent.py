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
    step: int = 0,
) -> Tuple[Dict[str, Any], str]:
    prompt = _build_decision_prompt(
        message, history, tools, tool_results, force_final=force_final
    )
    logger.info(
        "Agent asking Gemini for next action step=%d force_final=%s "
        "tool_count=%d tool_results_count=%d prompt_length=%d",
        step,
        force_final,
        len(tools),
        len(tool_results),
        len(prompt),
    )
    logger.debug("Agent decision prompt step=%d:\n%s", step, prompt)
    raw, model = await generate_chat_reply_async(prompt)
    logger.info(
        "Agent Gemini raw reply step=%d model=%r reply_length=%d preview=%r",
        step,
        model,
        len(raw),
        raw[:300],
    )
    try:
        action = _extract_json(raw)
        logger.info("Agent parsed action step=%d action=%s", step, action)
        return action, model
    except (json.JSONDecodeError, ValueError) as first_exc:
        logger.warning(
            "Agent JSON parse failed step=%d; retrying once: %s raw_preview=%r",
            step,
            first_exc,
            raw[:300],
        )
        repair_prompt = (
            "Your previous reply was not valid JSON.\n"
            "Previous reply:\n{}\n\n"
            "Return ONLY a valid JSON object for the next action "
            "(tool or final), no markdown."
        ).format(raw)
        raw2, model = await generate_chat_reply_async(repair_prompt)
        logger.info(
            "Agent Gemini repair reply step=%d model=%r reply_length=%d preview=%r",
            step,
            model,
            len(raw2),
            raw2[:300],
        )
        try:
            action = _extract_json(raw2)
            logger.info("Agent parsed repaired action step=%d action=%s", step, action)
            return action, model
        except (json.JSONDecodeError, ValueError) as second_exc:
            logger.error(
                "Agent JSON parse failed twice step=%d error=%s raw=%r repair_raw=%r",
                step,
                second_exc,
                raw,
                raw2,
            )
            raise AgentError(
                "Gemini did not return valid tool/final JSON: {}".format(second_exc)
            ) from second_exc


async def run_tool_agent(
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    """Run the async tool-selection loop and return (reply, model)."""
    logger.info(
        "Agent run_tool_agent starting message=%r history_turns=%d",
        message,
        len(history or []),
    )
    tools = await list_tool_definitions()
    logger.info(
        "Agent loaded tools count=%d names=%s",
        len(tools),
        [tool.get("name") for tool in tools],
    )
    if not tools:
        logger.error("Agent aborting because MCP returned no tools")
        raise AgentError("arun-mcp-server returned no tools.")

    tool_results: List[Dict[str, Any]] = []
    model = ""

    for step in range(MAX_AGENT_STEPS):
        force_final = step == MAX_AGENT_STEPS - 1
        logger.info(
            "Agent loop step=%d/%d force_final=%s",
            step + 1,
            MAX_AGENT_STEPS,
            force_final,
        )
        action, model = await _ask_for_action(
            message,
            history,
            tools,
            tool_results,
            force_final=force_final,
            step=step + 1,
        )
        action_type = str(action.get("action", "")).strip().lower()
        logger.info(
            "Agent step=%s resolved action=%s payload=%s",
            step + 1,
            action_type,
            action,
        )

        if action_type == "final":
            reply = str(action.get("reply", "")).strip()
            if not reply:
                logger.error("Agent final action had empty reply step=%d", step + 1)
                raise AgentError("Gemini returned a final action with an empty reply.")
            logger.info(
                "Agent completed with final reply step=%d model=%r reply_length=%d",
                step + 1,
                model,
                len(reply),
            )
            return reply, model

        if action_type == "tool":
            if force_final:
                logger.error(
                    "Agent hit max steps without final reply tool_results=%s",
                    tool_results,
                )
                raise AgentError("Max agent steps reached without a final reply.")
            name = str(action.get("name", "")).strip()
            args = action.get("args") or {}
            if not isinstance(args, dict):
                logger.error(
                    "Agent tool args invalid step=%d name=%r args_type=%s args=%r",
                    step + 1,
                    name,
                    type(args).__name__,
                    args,
                )
                raise AgentError("Tool args must be a JSON object.")
            logger.info(
                "Agent executing tool step=%d name=%r args=%s",
                step + 1,
                name,
                args,
            )
            try:
                result = await execute_tool(name, args)
            except ToolExecutionError as exc:
                logger.warning(
                    "Agent tool execution failed step=%d name=%r error=%s",
                    step + 1,
                    name,
                    exc,
                )
                result = {"error": str(exc)}
            tool_results.append({"name": name, "args": args, "result": result})
            logger.info(
                "Agent stored tool result step=%d total_results=%d result_preview=%r",
                step + 1,
                len(tool_results),
                str(result)[:300],
            )
            continue

        logger.error(
            "Agent unknown action step=%d action_type=%r payload=%s",
            step + 1,
            action_type,
            action,
        )
        raise AgentError(
            "Gemini returned unknown action {!r}. Expected 'tool' or 'final'.".format(
                action_type
            )
        )

    logger.error("Agent loop exited without final reply tool_results=%s", tool_results)
    raise AgentError("Agent loop ended without a final reply.")
