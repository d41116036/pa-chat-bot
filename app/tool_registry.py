"""MCP client that discovers and runs tools from arun-mcp-server."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp import Client
from mcp.types import TextContent

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


def _tool_input_schema(tool: Any) -> Dict[str, Any]:
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    if isinstance(schema, dict):
        return schema
    return {}


async def list_tool_definitions() -> List[Dict[str, Any]]:
    """Ask arun-mcp-server which tools are available (dynamic)."""
    try:
        async with Client(_mcp_server_url()) as client:
            result = await client.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": _tool_input_schema(tool),
                }
                for tool in result.tools
            ]
    except Exception as exc:
        raise ToolExecutionError(
            "Failed to list tools from arun-mcp-server at {}: {}".format(
                _mcp_server_url(), exc
            )
        ) from exc


def _result_payload(result: Any) -> Any:
    structured = getattr(result, "structured_content", None)
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
    try:
        async with Client(_mcp_server_url()) as client:
            result = await client.call_tool(name, args or {})
    except Exception as exc:
        raise ToolExecutionError(
            "Failed to call tool {!r} on arun-mcp-server: {}".format(name, exc)
        ) from exc

    if getattr(result, "is_error", False):
        raise ToolExecutionError(
            "Tool {!r} returned an error: {}".format(name, _result_payload(result))
        )
    return _result_payload(result)
