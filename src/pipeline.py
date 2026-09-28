"""
High-level convenience API: the single place most callers need.

    from src.pipeline import ingest_directory, ingest_file, Session

    ingest_directory("data/sample_docs")
    s = Session()
    result = s.ask("How do I reset my password?")
    print(result["answer"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.ingestion import chunk_documents, load_directory, load_file
from src.vectorstore import add_chunks, delete_source

__all__ = ["ingest_directory", "ingest_file", "Session"]


def __getattr__(name):
    # Lazy re-export so ingestion-only callers (MCP server, reindex watcher)
    # don't import the LLM stack.
    if name == "Session":
        from src.agent import Session

        return Session
    raise AttributeError(name)


def _replace_source(docs) -> int:
    """Delete a source's old chunks, then index the new ones.

    Without the delete, editing a file would leave its previous chunks in
    the store and the agent could cite text that no longer exists.
    """
    chunks = chunk_documents(docs)
    for source in {d.metadata.get("source") for d in docs}:
        delete_source(source)
    return add_chunks(chunks)


def ingest_directory(directory: str | Path) -> Dict[str, int]:
    """Load, chunk, and index every supported file under `directory`.

    Returns a dict: {"files": N, "chunks": M}.
    """
    docs = load_directory(directory)
    added = _replace_source(docs) if docs else 0
    files = len({d.metadata.get("source") for d in docs})
    return {"files": files, "chunks": added}


def ingest_file(path: str | Path) -> Dict[str, int]:
    """Load, chunk, and index a single file (replacing any earlier version)."""
    docs = load_file(path)
    if not docs:
        return {"files": 0, "chunks": 0}
    return {"files": 1, "chunks": _replace_source(docs)}


def sync_directory(directory: str | Path, manifest_path: str | Path) -> Dict[str, list]:
    """Bring the index in line with a folder: the automated re-index step.

    Compares each file's SHA-1 against a manifest from the previous run and
    re-ingests only new or changed files; files that disappeared are
    removed from the index. Returns {"added": [...], "updated": [...], "removed": [...]}.
    """
    import hashlib
    import json

    from src.ingestion.loaders import _LOADERS

    directory, manifest_path = Path(directory), Path(manifest_path)
    previous: Dict[str, str] = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    paths = {p.name: p for p in sorted(directory.rglob("*")) if p.is_file() and p.suffix.lower() in _LOADERS}
    current = {name: hashlib.sha1(p.read_bytes()).hexdigest() for name, p in paths.items()}

    report: Dict[str, list] = {"added": [], "updated": [], "removed": []}
    for name, digest in current.items():
        if previous.get(name) == digest:
            continue
        ingest_file(paths[name])
        report["updated" if name in previous else "added"].append(name)
    for name in previous.keys() - current.keys():
        delete_source(name)
        report["removed"].append(name)

    manifest_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return report
