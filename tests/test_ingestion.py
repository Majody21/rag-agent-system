"""
Tests for the ingestion layer. Pure-function tests with no API calls.
"""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ingestion import chunk_documents, load_file
from src.ingestion.loaders import load_csv, load_markdown


SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def test_load_markdown_preserves_section_headings(tmp_path):
    md = tmp_path / "doc.md"
    md.write_text(
        "# Title\n\nIntro text.\n\n## Section A\n\nBody A.\n\n## Section B\n\nBody B.\n",
        encoding="utf-8",
    )
    docs = load_markdown(md)
    headings = [d.metadata["section_heading"] for d in docs]
    assert "Title" in headings
    assert "Section A" in headings
    assert "Section B" in headings
    assert all(d.metadata["filetype"] == "markdown" for d in docs)


def test_load_csv_one_doc_per_row(tmp_path):
    csv = tmp_path / "people.csv"
    csv.write_text("name,role\nAlice,Engineer\nBob,Analyst\n", encoding="utf-8")
    docs = load_csv(csv)
    assert len(docs) == 2
    assert "Alice" in docs[0].page_content
    assert docs[0].metadata["row"] == 1
    assert docs[1].metadata["row"] == 2


def test_load_file_returns_empty_for_unknown_extension(tmp_path):
    junk = tmp_path / "binary.xyz"
    junk.write_bytes(b"\x00\x01\x02")
    assert load_file(junk) == []


def test_chunk_documents_preserves_metadata():
    docs = [
        Document(
            page_content="A" * 3000,
            metadata={"source": "big.md", "filetype": "markdown"},
        )
    ]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    # All chunks inherit source metadata + get chunk_index
    assert all(c.metadata["source"] == "big.md" for c in chunks)
    assert all("chunk_index" in c.metadata for c in chunks)
    # chunk_index is sequential within a source
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == sorted(indices)


def test_sample_docs_directory_exists():
    """Smoke test that the repo ships with sample docs."""
    assert SAMPLE_DIR.exists(), f"Expected {SAMPLE_DIR} to exist"
    assert any(SAMPLE_DIR.iterdir()), "sample_docs is empty"
