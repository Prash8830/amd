"""Thin synchronous MCP client for the telecom enterprise server.

The orchestrator is synchronous; MCP's Python SDK is async — each call wraps
its own event loop. Escalations and outage checks are infrequent, so the
per-call session cost (~50-150 ms) is acceptable and keeps the client
stateless and crash-proof.

Availability is probed once (cheap) so a missing server never adds latency
to the hot path.
"""

from __future__ import annotations
import asyncio
import json
import os

MCP_URL = os.environ.get("TELECOM_MCP_URL", "http://localhost:8765/sse")
_TIMEOUT_S = 3.0


def _parse_tool_result(result) -> dict | str | None:
    structured = getattr(result, "structuredContent", None)
    if structured:
        # FastMCP wraps plain returns as {"result": ...}
        return structured.get("result", structured)
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return text
    return None


async def _call_async(tool: str, args: dict):
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            return _parse_tool_result(result)


def call_mcp_tool(tool: str, args: dict | None = None):
    """Call a tool on the telecom MCP server. Returns None on any failure."""
    try:
        return asyncio.run(
            asyncio.wait_for(_call_async(tool, args or {}), timeout=_TIMEOUT_S))
    except Exception:
        return None


def mcp_available() -> bool:
    """One-shot availability probe (used at orchestrator startup)."""
    return call_mcp_tool("get_current_datetime") is not None
