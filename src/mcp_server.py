"""
MCP server exposing the knowledge base as tools.

    python src/mcp_server.py            # stdio transport (Claude Desktop, the agent)
    mcp dev src/mcp_server.py           # open in the MCP Inspector

Tools:
    search_documents(query, source_filter?, k?)  -> passages with citation tags
    list_sources()                               -> indexed file names
    get_document(source)                         -> every chunk of one file, in order
    ingest_document(path)                        -> index a file under RAG_DOCS_DIR

This process is the single owner of the vector store: the chat agent, the
Streamlit upload, and any MCP host all reach the index through these tools.

Nothing may be printed to stdout here: stdout is the MCP transport.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

import config  # noqa: E402
from src.ingestion.loaders import _LOADERS  # noqa: E402
from src.pipeline import ingest_file  # noqa: E402
from src.retrieval import format_sources, retrieve  # noqa: E402
from src.vectorstore import get_source_chunks  # noqa: E402
from src.vectorstore import list_sources as _list_sources  # noqa: E402

mcp = FastMCP(
    "rag-knowledge-base",
    log_level="WARNING",
    instructions=(
        "Company knowledge base. Call search_documents before answering factual "
        "questions and cite passages with their [source=...] tags."
    ),
)


@mcp.tool()
def search_documents(query: str, source_filter: str | None = None, k: int = config.RETRIEVAL_K) -> str:
    """Search the knowledge base and return the most relevant passages.

    Each passage starts with a tag like [source=it_password_reset.pdf, page=2, chunk=4]
    that identifies exactly where it came from. Cite those tags in answers.

    Args:
        query: A focused, keyword-rich search query, e.g. "password reset procedure".
        source_filter: Optional file name (e.g. "hr_onboarding.md") to search only that document.
        k: Number of passages to return (1-10).
    """
    filters = {"source": source_filter} if source_filter else None
    return format_sources(retrieve(query, k=max(1, min(int(k), 10)), filters=filters))


@mcp.tool()
def list_sources() -> str:
    """List the file names of every document currently indexed."""
    sources = _list_sources()
    if not sources:
        return "No documents are currently indexed."
    return f"Indexed documents ({len(sources)}):\n" + "\n".join(f"- {s}" for s in sources)


@mcp.tool()
def get_document(source: str) -> str:
    """Return every chunk of one indexed document, in order, for summaries or overviews.

    Args:
        source: Exact file name as shown by list_sources, e.g. "finance_expense_policy.md".
    """
    chunks = get_source_chunks(source)
    if not chunks:
        return f"No document named '{source}' is indexed. Use list_sources to see available names."
    return f"[full contents of {source}]\n\n" + "\n\n".join(c.page_content for c in chunks)


def _resolve_allowed(path: str) -> Path:
    """Only files inside RAG_DOCS_DIR with a supported extension may be ingested.

    Without this, a prompt could ask the agent to index any file on the
    machine (for example .env) and then read it back through search.
    """
    root = config.docs_dir().resolve()
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Only files inside {root} can be ingested.")
    if resolved.suffix.lower() not in _LOADERS:
        raise ValueError(f"Unsupported file type '{resolved.suffix}'. Supported: {sorted(_LOADERS)}")
    if not resolved.is_file():
        raise ValueError(f"File not found: {resolved}")
    return resolved


@mcp.tool()
def ingest_document(path: str) -> str:
    """Index (or re-index) one document so it becomes searchable.

    Re-ingesting a file replaces its previous chunks. Supported types: PDF,
    Markdown, CSV, TXT. The file must be inside the documents folder.

    Args:
        path: File path, relative to the documents folder or absolute inside it.
    """
    resolved = _resolve_allowed(path)
    result = ingest_file(resolved)
    return f"Ingested {resolved.name}: {result['chunks']} chunks indexed."


if __name__ == "__main__":
    # Set here, not at import: changing the environment on import would alter
    # Chroma settings for any process that merely imports this module.
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    mcp.run()  # stdio
