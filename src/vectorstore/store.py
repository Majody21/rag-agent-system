"""
Vector store wrapper (ChromaDB, persistent).

Key guarantees:
- Idempotent upserts: content-hash IDs prevent duplicates on re-ingestion.
- Single persistent client shared across the process.
- Swapping to Pinecone / OpenSearch is a one-file change.
"""

from __future__ import annotations

import hashlib
import shutil
from functools import lru_cache
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR, EMBED_MODEL, check_api_keys


def _chunk_id(doc: Document) -> str:
    """Deterministic ID from source + content hash. Same chunk → same ID."""
    source = doc.metadata.get("source", "unknown")
    h = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{source}::{h}"


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Return the persistent Chroma vectorstore. Cached for the process."""
    check_api_keys()
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )


def add_chunks(chunks: List[Document]) -> int:
    """
    Upsert chunks into the vector store. Returns count of unique chunks added.
    Duplicates (by content hash) are skipped silently.
    """
    if not chunks:
        return 0
    vs = get_vectorstore()
    ids = [_chunk_id(c) for c in chunks]

    # Dedupe within this batch (a file re-submitted in the same run)
    seen: set[str] = set()
    unique_chunks: List[Document] = []
    unique_ids: List[str] = []
    for doc, cid in zip(chunks, ids):
        if cid in seen:
            continue
        seen.add(cid)
        unique_chunks.append(doc)
        unique_ids.append(cid)

    # Chroma's add_documents with explicit IDs is effectively an upsert —
    # passing the same ID twice overwrites rather than duplicates.
    vs.add_documents(unique_chunks, ids=unique_ids)
    return len(unique_chunks)


def reset_store() -> None:
    """Nuke the entire persistent directory. Clears the lru_cache too."""
    get_vectorstore.cache_clear()
    if CHROMA_PERSIST_DIR.exists():
        shutil.rmtree(CHROMA_PERSIST_DIR)


def list_sources() -> List[str]:
    """Return the distinct `source` filenames currently indexed."""
    vs = get_vectorstore()
    try:
        # Chroma's underlying collection exposes .get() for raw metadata access
        result = vs._collection.get(include=["metadatas"])
        sources = {m.get("source") for m in (result.get("metadatas") or []) if m}
        return sorted(s for s in sources if s)
    except Exception:
        return []
