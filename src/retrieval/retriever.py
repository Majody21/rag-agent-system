"""
Retrieval: thin wrapper over the vector store facade (Chroma or OpenSearch).

Exposes two functions:
- retrieve(query, k, filters) → List[Document]
- format_sources(docs)        → human-readable citation string

We keep this separate from the vectorstore module so the agent's tool
layer can reason in terms of *retrieval intent* rather than storage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from config import RETRIEVAL_K
from src.vectorstore import similarity_search


def retrieve(
    query: str,
    k: int = RETRIEVAL_K,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Similarity search. Pass `filters={"source": "hr_onboarding.md"}` to
    restrict results to a specific document or filetype.
    """
    return similarity_search(query, k=k, filters=filters)


def format_sources(docs: List[Document]) -> str:
    """
    Render retrieved docs as a markdown-ish string for the LLM context,
    with clear [source: X, chunk N] tags so Claude can cite precisely.
    """
    if not docs:
        return "(no relevant sources found)"

    blocks: List[str] = []
    for d in docs:
        md = d.metadata
        source = md.get("source", "unknown")
        tags = [f"source={source}"]
        if "page" in md:
            tags.append(f"page={md['page']}")
        if "section_heading" in md and md.get("section_heading"):
            tags.append(f"section='{md['section_heading']}'")
        if "chunk_index" in md:
            tags.append(f"chunk={md['chunk_index']}")
        tag_str = ", ".join(tags)
        blocks.append(f"[{tag_str}]\n{d.page_content.strip()}")

    return "\n\n---\n\n".join(blocks)
