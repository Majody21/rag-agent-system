"""
High-level convenience API — the single place most callers need.

    from src.pipeline import ingest_directory, ingest_file, Session

    ingest_directory("data/sample_docs")
    s = Session()
    result = s.ask("How do I reset my password?")
    print(result["answer"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.agent import Session  # re-export
from src.ingestion import chunk_documents, load_directory, load_file
from src.vectorstore import add_chunks

__all__ = ["ingest_directory", "ingest_file", "Session"]


def ingest_directory(directory: str | Path) -> Dict[str, int]:
    """Load, chunk, and index every supported file under `directory`.

    Returns a dict: {"files": N, "chunks": M}.
    """
    docs = load_directory(directory)
    chunks = chunk_documents(docs)
    added = add_chunks(chunks)

    # Count distinct source files
    files = len({d.metadata.get("source") for d in docs})
    return {"files": files, "chunks": added}


def ingest_file(path: str | Path) -> Dict[str, int]:
    """Load, chunk, and index a single file."""
    docs = load_file(path)
    if not docs:
        return {"files": 0, "chunks": 0}
    chunks = chunk_documents(docs)
    added = add_chunks(chunks)
    return {"files": 1, "chunks": added}
