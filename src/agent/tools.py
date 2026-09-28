"""
Agent tools. Three capabilities exposed to the LLM:

- search_knowledge_base : RAG retrieval over all docs (primary)
- list_documents        : enumerate what's indexed
- get_document_summary  : fetch all chunks for one doc (for summarization)

Each tool returns a plain string — the agent reads the string and decides
what to do with it. We keep tools small and orthogonal so the agent's
reasoning stays cheap and auditable.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from src.retrieval import format_sources, retrieve
from src.vectorstore import get_source_chunks, list_sources


@tool
def search_knowledge_base(query: str, source_filter: Optional[str] = None) -> str:
    """Search the company knowledge base for information relevant to the query.

    Returns retrieved document chunks with their source citations. Always use
    this tool before answering any factual question.

    Args:
        query: A focused, specific search query. Rephrase user questions into
               keyword-rich form if needed (e.g. "password reset procedure"
               rather than "how do I reset my password").
        source_filter: Optional. If the user has specified a particular
                       document (e.g. "in the HR handbook"), pass the filename
                       like 'hr_onboarding.md' to restrict the search.
    """
    filters = {"source": source_filter} if source_filter else None
    docs = retrieve(query, filters=filters)
    return format_sources(docs)


@tool
def list_documents() -> str:
    """List the filenames of all documents currently indexed in the knowledge base.

    Use this when the user asks what information is available, what sources
    exist, or what the system knows about.
    """
    sources = list_sources()
    if not sources:
        return "No documents are currently indexed."
    lines = [f"- {s}" for s in sources]
    return f"Indexed documents ({len(sources)}):\n" + "\n".join(lines)


@tool
def get_document_summary(source: str) -> str:
    """Retrieve all chunks from a specific document so you can summarize it.

    Use this when the user asks for a summary or overview of a specific doc.
    Pass the exact filename (e.g. 'finance_expense_policy.md').
    """
    try:
        chunks = get_source_chunks(source)
    except Exception as e:
        return f"Error fetching document '{source}': {e}"
    if not chunks:
        return f"No document named '{source}' is indexed. Use list_documents to see available names."
    body = "\n\n".join(c.page_content for c in chunks)
    return f"[full contents of {source}]\n\n{body}"
