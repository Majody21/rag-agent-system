"""
Central configuration for the RAG Agent system.

All tunable knobs live here so the rest of the codebase stays clean.
Override any value via environment variable (see .env.example).

Providers:
- LLM: Anthropic Claude (ANTHROPIC_API_KEY)
- Embeddings: Google Gemini embeddings (GOOGLE_API_KEY)
- Vector store: local ChromaDB by default, or AWS OpenSearch Serverless
  when VECTOR_BACKEND=opensearch
"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ─── Paths ───────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
EVAL_DIR = PROJECT_ROOT / "eval"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"


def chroma_persist_dir() -> Path:
    """Read at call time so tests (and the demo) can redirect the store."""
    return Path(os.getenv("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR))


def docs_dir() -> Path:
    """The only folder the ingest_document MCP tool may read from."""
    return Path(os.getenv("RAG_DOCS_DIR", DATA_DIR))


UPLOADS_SUBDIR = "uploads"  # Streamlit uploads are saved under docs_dir()/uploads


# ─── Models ──────────────────────────────────────────────────────────
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-opus-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", AGENT_MODEL)
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "16000"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")

# ─── Chunking ────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ─── Retrieval ───────────────────────────────────────────────────────
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

# ─── Vector store ────────────────────────────────────────────────────
# "chroma" (local, default) or "opensearch" (AWS OpenSearch Serverless)
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma").lower()
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "enterprise_knowledge")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT", "")  # https://<id>.<region>.aoss.amazonaws.com
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "enterprise-knowledge")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# ─── Agent ───────────────────────────────────────────────────────────
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "5"))
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "6"))


def check_llm_key() -> None:
    """Raise a friendly error if the Claude key is missing."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "Missing ANTHROPIC_API_KEY. Create one at "
            "https://console.anthropic.com/settings/keys, then copy "
            ".env.example to .env and paste the key in."
        )


def check_embedding_key() -> None:
    """Raise a friendly error if the embeddings key is missing."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Missing GOOGLE_API_KEY (used for embeddings). Get one at "
            "https://aistudio.google.com/app/apikey and add it to .env."
        )
