"""
Retrieval tests — require live OpenAI embeddings, so marked `live`.
"""

import pytest

from src.pipeline import ingest_directory
from src.retrieval import format_sources, retrieve
from src.vectorstore import list_sources, reset_store
from config import SAMPLE_DOCS_DIR


@pytest.mark.live
def test_retrieval_roundtrip(tmp_path, monkeypatch):
    """
    Ingest the sample docs into a throwaway Chroma dir, then verify
    that a known password-reset question surfaces the IT SOP.
    """
    # Redirect Chroma to a tmp dir so we don't pollute the real store
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    # Force cache reset — the vectorstore module caches the client
    from src.vectorstore import store as store_mod

    store_mod.get_vectorstore.cache_clear()
    # Also reset config's path cache
    import importlib
    import config
    importlib.reload(config)

    reset_store()
    result = ingest_directory(SAMPLE_DOCS_DIR)
    assert result["chunks"] > 0
    assert "it_password_reset.pdf" in list_sources() or "it_password_reset_source.md" in list_sources()

    docs = retrieve("how do I reset my password", k=4)
    assert docs, "expected at least one retrieval hit"
    sources = {d.metadata.get("source") for d in docs}
    # At least one should be an IT doc
    assert any("password" in (s or "").lower() for s in sources)

    rendered = format_sources(docs)
    assert "source=" in rendered
