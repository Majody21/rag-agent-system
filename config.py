"""
Central configuration for the RAG Agent system.

All tunable knobs live here so the rest of the codebase stays clean.
Override any value via environment variable (see .env.example).

This build uses Google Gemini (free tier) end-to-end — one API key only.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
EVAL_DIR = PROJECT_ROOT / "eval"
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", PROJECT_ROOT / "chroma_db"))

# ─── Models ──────────────────────────────────────────────────────────
AGENT_MODEL = os.getenv("AGENT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")

# ─── Chunking ────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ─── Retrieval ───────────────────────────────────────────────────────
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

# ─── Collection ──────────────────────────────────────────────────────
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "enterprise_knowledge")

# ─── Agent ───────────────────────────────────────────────────────────
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.1"))
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "5"))
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "6"))


def check_api_keys() -> None:
    """Raise a friendly error if required keys are missing."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Get a free key at "
            "https://aistudio.google.com/app/apikey, then copy "
            ".env.example to .env and paste the key in."
        )
