"""
Retrieval tests.

- Offline tests exercise the vector store facade with fake embeddings.
- The `live` test runs real embeddings against the sample docs.
"""

import pytest
from langchain_core.documents import Document

from src.pipeline import ingest_file
from src.retrieval import format_sources, retrieve
from src.vectorstore import add_chunks, delete_source, get_source_chunks, list_sources, reset_store
from src.vectorstore.opensearch_backend import index_body, knn_query


def _doc(text, source="a.md", idx=0):
    return Document(page_content=text, metadata={"source": source, "filetype": "markdown", "chunk_index": idx})


def test_upsert_is_idempotent(offline_store):
    chunks = [_doc("alpha"), _doc("beta", idx=1)]
    add_chunks(chunks)
    add_chunks(chunks)
    assert len(get_source_chunks("a.md")) == 2


def test_reingest_replaces_stale_chunks(offline_store, tmp_path):
    f = tmp_path / "policy.md"
    f.write_text("# Limits\n\nThe meal limit is $50 per day.\n", encoding="utf-8")
    ingest_file(f)
    f.write_text("# Limits\n\nThe meal limit is $75 per day.\n", encoding="utf-8")
    ingest_file(f)
    texts = " ".join(c.page_content for c in get_source_chunks("policy.md"))
    assert "$75" in texts and "$50" not in texts


def test_delete_and_reset(offline_store):
    add_chunks([_doc("one", "x.md"), _doc("two", "y.md")])
    assert list_sources() == ["x.md", "y.md"]
    assert delete_source("x.md") == 1
    assert list_sources() == ["y.md"]
    reset_store()
    assert list_sources() == []


def test_format_sources_tags_citations():
    doc = Document(page_content="Body", metadata={"source": "it.pdf", "page": 2, "chunk_index": 5})
    rendered = format_sources([doc])
    assert rendered.startswith("[source=it.pdf, page=2, chunk=5]")


def test_opensearch_query_shapes():
    body = index_body(3072)
    assert body["mappings"]["properties"]["embedding"]["dimension"] == 3072
    q = knn_query([0.1, 0.2], k=4, filters={"source": "hr_onboarding.md"})
    knn = q["query"]["knn"]["embedding"]
    assert knn["k"] == 4 and q["size"] == 4
    assert knn["filter"]["bool"]["must"] == [{"term": {"metadata.source": "hr_onboarding.md"}}]


@pytest.mark.live
def test_retrieval_roundtrip(live_store):
    assert "it_password_reset.pdf" in list_sources()
    docs = retrieve("how do I reset my password", k=4)
    assert docs, "expected at least one retrieval hit"
    assert any("password" in (d.metadata.get("source") or "").lower() for d in docs)
    assert "source=" in format_sources(docs)


def test_sync_directory_detects_add_change_remove(offline_store, tmp_path):
    from src.pipeline import sync_directory

    docs, manifest = tmp_path / "docs", tmp_path / "manifest.json"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nfirst version\n", encoding="utf-8")
    (docs / "b.md").write_text("# B\n\nkeep me\n", encoding="utf-8")
    assert sorted(sync_directory(docs, manifest)["added"]) == ["a.md", "b.md"]

    assert sync_directory(docs, manifest) == {"added": [], "updated": [], "removed": []}

    (docs / "a.md").write_text("# A\n\nsecond version\n", encoding="utf-8")
    (docs / "b.md").unlink()
    report = sync_directory(docs, manifest)
    assert report["updated"] == ["a.md"] and report["removed"] == ["b.md"]
    assert list_sources() == ["a.md"]
    assert "second version" in get_source_chunks("a.md")[0].page_content
