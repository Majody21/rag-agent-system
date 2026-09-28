"""
Pytest configuration — makes the project root importable and marks
tests that need live API keys so they're skipped in CI by default.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked `live` when the API key is missing."""
    if os.getenv("GOOGLE_API_KEY") and os.getenv("ANTHROPIC_API_KEY"):
        return
    skip_live = pytest.mark.skip(reason="GOOGLE_API_KEY and ANTHROPIC_API_KEY required for live tests")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires live API keys (Anthropic + Google embeddings)")


def _isolate_store(mp, persist_dir, fake_embeddings: bool):
    """Point the vector store at a throwaway Chroma dir (never the real index)."""
    from src.vectorstore import store as store_mod

    mp.setenv("CHROMA_PERSIST_DIR", str(persist_dir))
    mp.setattr("config.VECTOR_BACKEND", "chroma")
    if fake_embeddings:
        from langchain_core.embeddings import DeterministicFakeEmbedding

        mp.setattr(store_mod, "get_embeddings", lambda: DeterministicFakeEmbedding(size=64))
    store_mod.get_backend.cache_clear()


@pytest.fixture
def offline_store(tmp_path, monkeypatch):
    """Chroma in tmp_path with fake embeddings: no API keys, no network."""
    from src.vectorstore import store as store_mod

    _isolate_store(monkeypatch, tmp_path / "chroma", fake_embeddings=True)
    yield
    store_mod.get_backend.cache_clear()


@pytest.fixture(scope="session")
def live_store(tmp_path_factory):
    """Real embeddings, sample docs ingested ONCE per test run.

    Sharing one ingestion keeps the suite under the Gemini free-tier
    embedding quota (100 requests/minute).
    """
    from config import SAMPLE_DOCS_DIR
    from src.pipeline import ingest_directory
    from src.vectorstore import store as store_mod

    with pytest.MonkeyPatch.context() as mp:
        _isolate_store(mp, tmp_path_factory.mktemp("chroma"), fake_embeddings=False)
        ingest_directory(SAMPLE_DOCS_DIR)
        yield
        store_mod.get_backend.cache_clear()
