"""
Vector store facade.

Everything above this module (retriever, tools, ingestion, MCP server)
talks to these functions only, never to a specific database:

    add_chunks, similarity_search, list_sources, get_source_chunks,
    delete_source, reset_store

The backend is chosen by VECTOR_BACKEND:
- "chroma"     : persistent local ChromaDB (default, zero infra)
- "opensearch" : AWS OpenSearch Serverless vector collection
                 (see opensearch_backend.py)

Key guarantees:
- Idempotent upserts: content-hash IDs prevent duplicates on re-ingestion.
- Re-ingesting a changed file replaces its chunks (delete_source first),
  so stale passages cannot be cited.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

import config


def _chunk_id(doc: Document) -> str:
    """Deterministic ID from source + content hash. Same chunk -> same ID."""
    source = doc.metadata.get("source", "unknown")
    h = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{source}::{h}"


@lru_cache(maxsize=1)
def get_embeddings():
    """Embedding model shared by both backends."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    config.check_embedding_key()
    return GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL)


# ─── Chroma backend ──────────────────────────────────────────────────
class ChromaBackend:
    def __init__(self) -> None:
        from langchain_chroma import Chroma

        persist_dir = config.chroma_persist_dir()
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.vs = Chroma(
            collection_name=config.CHROMA_COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=str(persist_dir),
        )

    def upsert(self, docs: List[Document], ids: List[str]) -> None:
        # add_documents with explicit IDs overwrites rather than duplicates
        self.vs.add_documents(docs, ids=ids)

    def search(self, query: str, k: int, filters: Optional[Dict[str, Any]]) -> List[Document]:
        return self.vs.similarity_search(query, k=k, filter=filters or None)

    def list_sources(self) -> List[str]:
        result = self.vs._collection.get(include=["metadatas"])
        sources = {m.get("source") for m in (result.get("metadatas") or []) if m}
        return sorted(s for s in sources if s)

    def get_source_chunks(self, source: str) -> List[Document]:
        result = self.vs._collection.get(where={"source": source}, include=["documents", "metadatas"])
        return [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(result.get("documents") or [], result.get("metadatas") or [])
        ]

    def delete_source(self, source: str) -> int:
        existing = self.vs._collection.get(where={"source": source}, include=[])
        ids = existing.get("ids") or []
        if ids:
            self.vs._collection.delete(ids=ids)
        return len(ids)

    def reset(self) -> None:
        # Drop the collection through the client instead of deleting files:
        # on Windows the open SQLite handle makes rmtree fail.
        self.vs.delete_collection()


@lru_cache(maxsize=1)
def get_backend():
    """Return the configured backend. Cached for the process."""
    if config.VECTOR_BACKEND == "opensearch":
        from src.vectorstore.opensearch_backend import OpenSearchBackend

        return OpenSearchBackend(get_embeddings())
    if config.VECTOR_BACKEND != "chroma":
        raise RuntimeError(f"Unknown VECTOR_BACKEND '{config.VECTOR_BACKEND}' (use chroma or opensearch)")
    return ChromaBackend()


def get_vectorstore():
    """Backwards-compatible alias used by older callers."""
    return get_backend()


# ─── Public API ──────────────────────────────────────────────────────
def add_chunks(chunks: List[Document]) -> int:
    """Upsert chunks. Returns count of unique chunks written."""
    if not chunks:
        return 0
    seen: set[str] = set()
    unique_chunks: List[Document] = []
    unique_ids: List[str] = []
    for doc in chunks:
        cid = _chunk_id(doc)
        if cid in seen:
            continue
        seen.add(cid)
        unique_chunks.append(doc)
        unique_ids.append(cid)
    get_backend().upsert(unique_chunks, unique_ids)
    return len(unique_chunks)


def similarity_search(query: str, k: int, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
    return get_backend().search(query, k, filters)


def list_sources() -> List[str]:
    """Distinct `source` filenames currently indexed."""
    return get_backend().list_sources()


def get_source_chunks(source: str) -> List[Document]:
    """All chunks for one source, in document order."""
    docs = get_backend().get_source_chunks(source)
    return sorted(docs, key=lambda d: d.metadata.get("chunk_index", 0))


def delete_source(source: str) -> int:
    """Remove every chunk for one source. Returns how many were removed."""
    return get_backend().delete_source(source)


def reset_store() -> None:
    """Delete everything in the store and drop the cached client."""
    try:
        get_backend().reset()
    finally:
        get_backend.cache_clear()
