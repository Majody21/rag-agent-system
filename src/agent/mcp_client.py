"""
MCP client side: how the LangChain agent gets its tools.

The agent does not import the retrieval code. It launches the MCP server
(src/mcp_server.py) as a stdio subprocess, keeps ONE session open for the
life of the process, and converts the server's tools into LangChain tools
with langchain-mcp-adapters.

MCP tools are async, and the session is bound to the event loop that
opened it. So a background thread runs that loop, and synchronous callers
(Session.ask, Streamlit, the CLI) submit coroutines to it.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from contextlib import AsyncExitStack
from functools import lru_cache
from typing import Any, Dict, List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.stdio import get_default_environment

from config import PROJECT_ROOT

SERVER_NAME = "rag"
SERVER_SCRIPT = PROJECT_ROOT / "src" / "mcp_server.py"
# Least privilege: the chat agent can read the index but not add to it.
AGENT_TOOLS = ("search_documents", "list_sources", "get_document")
# Settings the server process needs. The Claude key is deliberately NOT passed.
_FORWARD_ENV_PREFIXES = ("GOOGLE_API_KEY", "EMBED_", "CHROMA_", "VECTOR_BACKEND", "OPENSEARCH_",
                         "AWS_", "RETRIEVAL_", "CHUNK_", "RAG_DOCS_DIR")


def server_connection() -> Dict[str, Any]:
    env = get_default_environment()
    env.update({k: v for k, v in os.environ.items() if k.startswith(_FORWARD_ENV_PREFIXES)})
    env["ANONYMIZED_TELEMETRY"] = "False"
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SERVER_SCRIPT)],
        "cwd": str(PROJECT_ROOT),
        "env": env,
    }


class MCPToolbox:
    """One long-lived MCP session plus the event loop it runs on."""

    def __init__(self, timeout: float = 120) -> None:
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, name="mcp-client", daemon=True).start()
        self.tools: List[BaseTool] = self.run(self._open(), timeout=timeout)
        self._by_name = {t.name: t for t in self.tools}

    async def _open(self) -> List[BaseTool]:
        self._stack = AsyncExitStack()
        client = MultiServerMCPClient({SERVER_NAME: server_connection()})
        self._session = await self._stack.enter_async_context(client.session(SERVER_NAME))
        return await load_mcp_tools(self._session)

    def run(self, coro, timeout: float | None = None):
        """Run a coroutine on the MCP loop from synchronous code."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    @property
    def agent_tools(self) -> List[BaseTool]:
        return [self._by_name[n] for n in AGENT_TOOLS]

    def call(self, name: str, **arguments: Any) -> str:
        """Call one MCP tool directly (used by the UI for uploads and listings)."""
        result = self.run(self._session.call_tool(name, arguments), timeout=600)
        text = "\n".join(getattr(c, "text", "") for c in result.content)
        if result.isError:
            raise RuntimeError(text)
        return text

    def close(self) -> None:
        self.run(self._stack.aclose(), timeout=30)
        self._loop.call_soon_threadsafe(self._loop.stop)


@lru_cache(maxsize=1)
def get_toolbox() -> MCPToolbox:
    """Process-wide toolbox: one MCP server subprocess shared by all sessions."""
    return MCPToolbox()


def list_sources() -> List[str]:
    """Indexed file names, fetched through the MCP list_sources tool."""
    text = get_toolbox().call("list_sources")
    return [line[2:] for line in text.splitlines() if line.startswith("- ")]
