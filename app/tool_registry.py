"""MCP client that discovers and runs tools from arun-mcp-server."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp import Client
from mcp.types import TextContent

logger = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parent.parent
_app_env = os.getenv("APP_ENV", "dev").lower()
_env_file = _project_root / ".env.{}".format(_app_env)
load_dotenv(
    dotenv_path=_env_file if _env_file.exists() else _project_root / ".env",
    override=True,
)

DEFAULT_ARUN_MCP_SERVER_URL = "http://127.0.0.1:9000/mcp"


class ToolExecutionError(Exception):
    """Raised when MCP tool discovery or execution fails."""


def _mcp_server_url() -> str:
    return os.getenv("ARUN_MCP_SERVER_URL", DEFAULT_ARUN_MCP_SERVER_URL).rstrip("/")


def _truncate_for_log(value: Any, max_len: int = 800) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= max_len:
        return text
    return "{}... (truncated, total {} chars)".format(text[:max_len], len(text))


def _tool_input_schema(tool: Any) -> Dict[str, Any]:
    schema = getattr(tool, "input_schema", None)
    schema_source = "input_schema"
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
        schema_source = "inputSchema"
    if isinstance(schema, dict):
        return schema
    if schema is not None and hasattr(schema, "model_dump"):
        dumped = schema.model_dump()
        logger.debug(
            "Converted MCP tool %r schema from %s via model_dump",
            getattr(tool, "name", "?"),
            schema_source,
        )
        return dumped
    if schema is not None:
        logger.warning(
            "MCP tool %r has unsupported schema type %s from %s; using empty parameters",
            getattr(tool, "name", "?"),
            type(schema).__name__,
            schema_source,
        )
    return {}


async def list_tool_definitions() -> List[Dict[str, Any]]:
    """Ask arun-mcp-server which tools are available (dynamic)."""
    url = _mcp_server_url()
    logger.info("MCP list_tool_definitions starting url=%s", url)
    try:
        async with Client(url) as client:
            result = await client.list_tools()
            raw_tools = getattr(result, "tools", None) or []
            logger.info(
                "MCP list_tools HTTP succeeded raw_tool_count=%d",
                len(raw_tools),
            )
            definitions: List[Dict[str, Any]] = []
            for index, tool in enumerate(raw_tools):
                definition = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": _tool_input_schema(tool),
                }
                definitions.append(definition)
                logger.debug(
                    "MCP tool[%d] name=%r description=%r parameters=%s",
                    index,
                    definition["name"],
                    definition["description"],
                    _truncate_for_log(definition["parameters"], max_len=400),
                )
            logger.info(
                "MCP list_tool_definitions parsed count=%d names=%s",
                len(definitions),
                [item["name"] for item in definitions],
            )
            if not definitions:
                logger.error(
                    "MCP list_tools returned zero tools after successful HTTP call url=%s",
                    url,
                )
            return definitions
    except Exception as exc:
        logger.exception(
            "MCP list_tool_definitions failed url=%s error=%s",
            url,
            exc,
        )
        raise ToolExecutionError(
            "Failed to list tools from arun-mcp-server at {}: {}".format(url, exc)
        ) from exc


def _result_payload(result: Any) -> Any:
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict) and "result" in structured:
        # Prefer the tool's returned value, not the wrapper object.
        return structured.get("result")
    if structured is not None:
        return structured
    texts: List[str] = []
    for block in getattr(result, "content", None) or []:
        if isinstance(block, TextContent):
            texts.append(block.text)
        else:
            text = getattr(block, "text", None)
            if text:
                texts.append(str(text))
    if len(texts) == 1:
        return texts[0]
    if texts:
        return texts
    return None


async def execute_tool(name: str, args: Optional[Dict[str, Any]] = None) -> Any:
    """Call a tool on arun-mcp-server by name."""
    url = _mcp_server_url()
    tool_args = args or {}
    logger.info(
        "MCP execute_tool starting url=%s name=%r args=%s",
        url,
        name,
        _truncate_for_log(tool_args, max_len=400),
    )
    try:
        async with Client(url) as client:
            result = await client.call_tool(name, tool_args)
    except Exception as exc:
        logger.exception(
            "MCP execute_tool failed url=%s name=%r args=%s error=%s",
            url,
            name,
            _truncate_for_log(tool_args, max_len=400),
            exc,
        )
        raise ToolExecutionError(
            "Failed to call tool {!r} on arun-mcp-server: {}".format(name, exc)
        ) from exc

    if getattr(result, "is_error", False):
        error_payload = _result_payload(result)
        logger.error(
            "MCP execute_tool returned is_error=true name=%r payload=%s",
            name,
            _truncate_for_log(error_payload),
        )
        raise ToolExecutionError(
            "Tool {!r} returned an error: {}".format(name, error_payload)
        )

    payload = _result_payload(result)
    logger.info(
        "MCP execute_tool succeeded name=%r result=%s",
        name,
        _truncate_for_log(payload),
    )
    return payload
