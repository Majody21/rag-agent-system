"""
MCP server tests: a real MCP client talks to the server over the SDK's
in-memory transport (same protocol as stdio, no subprocess). Offline:
uses the fake-embedding store from conftest.
"""

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

import config


def _run(coro_fn):
    return anyio.run(coro_fn)


async def _call(name, args=None):
    from src.mcp_server import mcp

    async with create_connected_server_and_client_session(mcp) as client:
        result = await client.call_tool(name, args or {})
        return result.isError, "\n".join(c.text for c in result.content)


def test_server_lists_expected_tools():
    from src.mcp_server import mcp

    async def go():
        async with create_connected_server_and_client_session(mcp) as client:
            return {t.name: t for t in (await client.list_tools()).tools}

    tools = _run(go)
    assert set(tools) == {"search_documents", "ingest_document", "list_sources", "get_document"}
    assert "query" in tools["search_documents"].inputSchema["required"]


@pytest.fixture
def docs_root(tmp_path, monkeypatch, offline_store):
    root = tmp_path / "docs"
    root.mkdir()
    monkeypatch.setenv("RAG_DOCS_DIR", str(root))
    (root / "policy.md").write_text("# Travel\n\nEconomy class for flights under 6 hours.\n", encoding="utf-8")
    return root


def test_ingest_then_search_returns_citation_tags(docs_root):
    is_err, text = _run(lambda: _call("ingest_document", {"path": "policy.md"}))
    assert not is_err and "policy.md" in text
    _, listing = _run(lambda: _call("list_sources"))
    assert "- policy.md" in listing
    _, hits = _run(lambda: _call("search_documents", {"query": "flight class", "k": 2}))
    assert "[source=policy.md" in hits and "Economy class" in hits
    _, full = _run(lambda: _call("get_document", {"source": "policy.md"}))
    assert full.startswith("[full contents of policy.md]")


@pytest.mark.parametrize("bad", ["../secret.txt", "C:/Windows/win.ini", "/etc/passwd.txt", "policy.exe"])
def test_ingest_rejects_paths_outside_docs_dir(docs_root, bad):
    (docs_root.parent / "secret.txt").write_text("API_KEY=xyz", encoding="utf-8")
    is_err, text = _run(lambda: _call("ingest_document", {"path": bad}))
    assert is_err
    _, listing = _run(lambda: _call("list_sources"))
    assert "secret" not in listing


def test_docs_dir_defaults_to_data(monkeypatch):
    monkeypatch.delenv("RAG_DOCS_DIR", raising=False)
    assert config.docs_dir() == config.DATA_DIR
