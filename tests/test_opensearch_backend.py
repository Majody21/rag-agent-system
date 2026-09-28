"""
OpenSearch Serverless adapter, tested against an in-memory fake client.

This checks the requests the adapter sends (index mapping, bulk upserts
with content-hash IDs, filtered k-NN, aggregations, deletes). It does NOT
prove the adapter against a live AWS collection; see README for that.
"""

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.vectorstore.opensearch_backend import OpenSearchBackend


class FakeIndices:
    def __init__(self):
        self.created = {}

    def exists(self, index):
        return index in self.created

    def create(self, index, body):
        self.created[index] = body

    def delete(self, index):
        self.created.pop(index, None)


class FakeClient:
    def __init__(self):
        self.indices = FakeIndices()
        self.docs = {}
        self.searches = []

    def bulk(self, body):
        for action, doc in zip(body[::2], body[1::2]) if "index" in body[0] else []:
            self.docs[action["index"]["_id"]] = doc
        for action in body:
            if "delete" in action:
                self.docs.pop(action["delete"]["_id"], None)
        return {"errors": False}

    def search(self, index, body):
        self.searches.append(body)
        if "aggs" in body:
            keys = sorted({d["metadata"]["source"] for d in self.docs.values()})
            return {"aggregations": {"sources": {"buckets": [{"key": k} for k in keys]}}}
        hits = [{"_id": i, "_source": d} for i, d in self.docs.items()]
        term = body["query"].get("term")
        if term:
            hits = [h for h in hits if h["_source"]["metadata"]["source"] == term["metadata.source"]]
        return {"hits": {"hits": hits[: body["size"]]}}


def _backend():
    return OpenSearchBackend(DeterministicFakeEmbedding(size=8), client=FakeClient(), index="kb")


def test_upsert_creates_index_and_uses_ids():
    b = _backend()
    b.upsert([Document(page_content="hi", metadata={"source": "a.md"})], ["a.md::abc"])
    mapping = b.client.indices.created["kb"]["mappings"]["properties"]
    assert mapping["embedding"]["dimension"] == 8
    assert "a.md::abc" in b.client.docs


def test_list_get_delete_reset():
    b = _backend()
    b.upsert(
        [Document(page_content="x", metadata={"source": "a.md"}), Document(page_content="y", metadata={"source": "b.md"})],
        ["a.md::1", "b.md::1"],
    )
    assert b.list_sources() == ["a.md", "b.md"]
    assert [d.page_content for d in b.get_source_chunks("b.md")] == ["y"]
    assert b.delete_source("a.md") == 1
    assert b.list_sources() == ["b.md"]
    b.reset()
    assert b.list_sources() == []


def test_search_sends_filtered_knn():
    b = _backend()
    b.upsert([Document(page_content="x", metadata={"source": "a.md"})], ["a.md::1"])
    b.search("question", k=3, filters={"source": "a.md"})
    knn = b.client.searches[-1]["query"]["knn"]["embedding"]
    assert knn["k"] == 3 and len(knn["vector"]) == 8
    assert knn["filter"]["bool"]["must"][0] == {"term": {"metadata.source": "a.md"}}
