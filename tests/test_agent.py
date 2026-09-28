"""
Agent smoke tests. Require live keys (Anthropic + Google embeddings).
"""

import pytest

from src.pipeline import Session


@pytest.mark.live
def test_agent_answers_with_citation(live_store):
    result = Session().ask("How do I reset my password if my MFA is working?")
    assert result["answer"]
    assert "search_knowledge_base" in result["tool_calls"]
    assert result["sources"], "expected at least one cited source"
    answer_lower = result["answer"].lower()
    assert any(kw in answer_lower for kw in ["forgot password", "login.acme", "okta", "mfa"])
    assert result["usage"], "token usage should be recorded"


@pytest.mark.live
def test_agent_lists_documents(live_store):
    result = Session().ask("What documents do you have access to?")
    assert "list_documents" in result["tool_calls"]
