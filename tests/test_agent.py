"""
End-to-end agent tests over MCP. Require live keys (Anthropic + Google
embeddings). The agent launches src/mcp_server.py as a stdio subprocess
and only reaches the knowledge base through its tools.
"""

import pytest

from src.pipeline import Session


@pytest.mark.live
def test_agent_answers_with_citation_via_mcp(live_store):
    from src.agent.mcp_client import get_toolbox

    assert [t.name for t in get_toolbox().agent_tools] == ["search_documents", "list_sources", "get_document"]
    result = Session().ask("How do I reset my password if my MFA is working?")
    assert "search_documents" in result["tool_calls"]
    assert any(s.get("source") == "it_password_reset.pdf" for s in result["sources"])
    assert "[source:" in result["answer"]
    answer_lower = result["answer"].lower()
    assert any(kw in answer_lower for kw in ["forgot password", "login.acme", "okta", "mfa"])
    assert result["usage"], "token usage should be recorded"


@pytest.mark.live
def test_agent_lists_documents_via_mcp(live_store):
    result = Session().ask("What documents do you have access to?")
    assert "list_sources" in result["tool_calls"]
