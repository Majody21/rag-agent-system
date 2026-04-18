"""
Chunking: splits loaded Documents into retrieval-sized chunks.

Uses LangChain's RecursiveCharacterTextSplitter — it tries to split on
paragraph → sentence → word → character boundaries in that order, so
semantic units are preserved when possible.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split each input Document into chunks of ~chunk_size chars with overlap.

    Metadata is preserved from the source Document, and each chunk gets a
    `chunk_index` (position within the source doc) added to its metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks: List[Document] = []
    # Group by source so chunk_index is per-source, not global
    by_source: dict[str, List[Document]] = {}
    for d in docs:
        by_source.setdefault(d.metadata.get("source", "unknown"), []).append(d)

    for source, source_docs in by_source.items():
        # Split each source-doc (which may itself be a page/section), then
        # assign a running chunk_index across the source.
        running_idx = 0
        for d in source_docs:
            pieces = splitter.split_documents([d])
            for piece in pieces:
                piece.metadata = {**d.metadata, **piece.metadata, "chunk_index": running_idx}
                running_idx += 1
                all_chunks.append(piece)

    return all_chunks
