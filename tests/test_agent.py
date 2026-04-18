"""
Agent smoke test — requires live keys (Anthropic + OpenAI).
"""

import pytest

from config import SAMPLE_DOCS_DIR
from src.pipeline import Session, ingest_directory
from src.vectorstore import list_sources


@pytest.mark.live
def test_agent_answers_with_citation(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    import importlib
    import config
    importlib.reload(config)
    from src.vectorstore import store as store_mod
    store_mod.get_vectorstore.cache_clear()

    # Seed the store
    ingest_directory(SAMPLE_DOCS_DIR)
    assert list_sources()

    session = Session()
    result = session.ask("How do I reset my password if my MFA is working?")
    assert result["answer"]
    # The agent should have called the retrieval tool at least once
    assert "search_knowledge_base" in result["tool_calls"]
    # Answer should mention something from the source
    answer_lower = result["answer"].lower()
    assert any(kw in answer_lower for kw in ["forgot password", "login.acme", "okta", "mfa"])


@pytest.mark.live
def test_agent_lists_documents(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    import importlib
    import config
    importlib.reload(config)
    from src.vectorstore import store as store_mod
    store_mod.get_vectorstore.cache_clear()

    ingest_directory(SAMPLE_DOCS_DIR)
    session = Session()
    result = session.ask("What documents do you have access to?")
    # Should have called the list_documents tool
    assert "list_documents" in result["tool_calls"]
