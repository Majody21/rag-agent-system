"""
AWS OpenSearch Serverless backend (vector search collection).

Enabled with VECTOR_BACKEND=opensearch plus OPENSEARCH_ENDPOINT and
AWS_REGION. Credentials come from the standard AWS chain (env vars,
~/.aws/credentials, or an IAM role); nothing is read from code.

Targets a NextGen vector collection, which supports user-provided
document IDs (so content-hash upserts stay idempotent) and scales to
zero OCUs when idle. Requests are signed with SigV4 for service "aoss".

Index layout (created on first write, dimension taken from the embedder):
    embedding : knn_vector (cosine)
    text      : the chunk text
    metadata  : source, filetype, page, section_heading, row, chunk_index
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from langchain_core.documents import Document

import config

_METADATA_MAPPING = {
    "properties": {
        "source": {"type": "keyword"},
        "filetype": {"type": "keyword"},
        "page": {"type": "integer"},
        "section_heading": {"type": "keyword"},
        "row": {"type": "integer"},
        "chunk_index": {"type": "integer"},
    }
}


def index_body(dimension: int) -> Dict[str, Any]:
    return {
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                "embedding": {"type": "knn_vector", "dimension": dimension, "space_type": "cosinesimil"},
                "text": {"type": "text"},
                "metadata": _METADATA_MAPPING,
            }
        },
    }


def knn_query(vector: List[float], k: int, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    knn: Dict[str, Any] = {"vector": vector, "k": k}
    if filters:
        knn["filter"] = {"bool": {"must": [{"term": {f"metadata.{f}": v}} for f, v in filters.items()]}}
    return {"size": k, "_source": ["text", "metadata"], "query": {"knn": {"embedding": knn}}}


def _make_client():
    import boto3
    from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

    if not config.OPENSEARCH_ENDPOINT:
        raise RuntimeError("VECTOR_BACKEND=opensearch but OPENSEARCH_ENDPOINT is not set.")
    credentials = boto3.Session().get_credentials()
    if credentials is None:
        raise RuntimeError("No AWS credentials found. Run `aws configure` or set an IAM role.")
    host = urlparse(config.OPENSEARCH_ENDPOINT).hostname or config.OPENSEARCH_ENDPOINT
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=AWSV4SignerAuth(credentials, config.AWS_REGION, "aoss"),
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
        timeout=60,
    )


class OpenSearchBackend:
    def __init__(self, embeddings, client=None, index: str = config.OPENSEARCH_INDEX) -> None:
        self.embeddings = embeddings
        self.client = client or _make_client()
        self.index = index

    def _ensure_index(self, dimension: int) -> None:
        if not self.client.indices.exists(index=self.index):
            self.client.indices.create(index=self.index, body=index_body(dimension))

    def upsert(self, docs: List[Document], ids: List[str]) -> None:
        vectors = self.embeddings.embed_documents([d.page_content for d in docs])
        self._ensure_index(len(vectors[0]))
        body: List[Dict[str, Any]] = []
        for doc, vec, doc_id in zip(docs, vectors, ids):
            body.append({"index": {"_index": self.index, "_id": doc_id}})
            body.append({"embedding": vec, "text": doc.page_content, "metadata": doc.metadata})
        resp = self.client.bulk(body=body)
        if resp.get("errors"):
            failed = [i for i in resp.get("items", []) if i.get("index", {}).get("error")]
            raise RuntimeError(f"OpenSearch bulk upsert failed for {len(failed)} chunk(s): {failed[:1]}")

    def _hits_to_docs(self, resp: Dict[str, Any]) -> List[Document]:
        return [
            Document(page_content=h["_source"]["text"], metadata=h["_source"].get("metadata", {}))
            for h in resp["hits"]["hits"]
        ]

    def search(self, query: str, k: int, filters: Optional[Dict[str, Any]]) -> List[Document]:
        if not self.client.indices.exists(index=self.index):
            return []
        vector = self.embeddings.embed_query(query)
        return self._hits_to_docs(self.client.search(index=self.index, body=knn_query(vector, k, filters)))

    def list_sources(self) -> List[str]:
        if not self.client.indices.exists(index=self.index):
            return []
        resp = self.client.search(
            index=self.index,
            body={"size": 0, "aggs": {"sources": {"terms": {"field": "metadata.source", "size": 1000}}}},
        )
        return sorted(b["key"] for b in resp["aggregations"]["sources"]["buckets"])

    def get_source_chunks(self, source: str) -> List[Document]:
        if not self.client.indices.exists(index=self.index):
            return []
        resp = self.client.search(
            index=self.index,
            body={
                "size": 1000,
                "_source": ["text", "metadata"],
                "query": {"term": {"metadata.source": source}},
            },
        )
        return self._hits_to_docs(resp)

    def delete_source(self, source: str) -> int:
        if not self.client.indices.exists(index=self.index):
            return 0
        # Look up IDs, then bulk-delete: plain search + bulk work on every collection type.
        resp = self.client.search(
            index=self.index,
            body={"size": 1000, "_source": False, "query": {"term": {"metadata.source": source}}},
        )
        ids = [h["_id"] for h in resp["hits"]["hits"]]
        if ids:
            self.client.bulk(body=[{"delete": {"_index": self.index, "_id": i}} for i in ids])
        return len(ids)

    def reset(self) -> None:
        if self.client.indices.exists(index=self.index):
            self.client.indices.delete(index=self.index)
