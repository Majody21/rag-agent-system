"""
Document loaders for PDF, Markdown, and CSV.

All loaders return a list of langchain `Document` objects with consistent
metadata: {source, filetype, page (PDF only), section_heading (MD only)}.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.documents import Document
from pypdf import PdfReader


# ─── PDF ─────────────────────────────────────────────────────────────
def load_pdf(path: Path) -> List[Document]:
    """Load a PDF. One Document per page, with page number in metadata."""
    reader = PdfReader(str(path))
    docs: List[Document] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "filetype": "pdf",
                    "page": i,
                },
            )
        )
    return docs


# ─── Markdown ────────────────────────────────────────────────────────
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def load_markdown(path: Path) -> List[Document]:
    """
    Load a markdown file. We split on top-level (## / ###) headings so each
    Document carries its section heading — this is the highest-leverage
    metadata for retrieval filtering.
    """
    text = path.read_text(encoding="utf-8")
    sections: List[tuple[str, str]] = []  # (heading, body)

    # Find all heading positions
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [
            Document(
                page_content=text,
                metadata={"source": path.name, "filetype": "markdown", "section_heading": ""},
            )
        ]

    # First chunk (before any heading) attaches to a synthetic "Intro"
    if matches[0].start() > 0:
        intro = text[: matches[0].start()].strip()
        if intro:
            sections.append(("Intro", intro))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append((heading, body))

    return [
        Document(
            page_content=f"# {h}\n\n{b}",
            metadata={"source": path.name, "filetype": "markdown", "section_heading": h},
        )
        for h, b in sections
    ]


# ─── CSV ─────────────────────────────────────────────────────────────
def load_csv(path: Path) -> List[Document]:
    """
    Load CSV as one Document per row. Row becomes a natural-language string
    like 'column_a: value | column_b: value' so embeddings work well.
    """
    df = pd.read_csv(path)
    docs: List[Document] = []
    for idx, row in df.iterrows():
        fields = " | ".join(f"{col}: {row[col]}" for col in df.columns)
        docs.append(
            Document(
                page_content=f"Row {idx + 1}: {fields}",
                metadata={
                    "source": path.name,
                    "filetype": "csv",
                    "row": int(idx) + 1,
                },
            )
        )
    return docs


# ─── Dispatcher ──────────────────────────────────────────────────────
_LOADERS = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".csv": load_csv,
    ".txt": lambda p: [
        Document(
            page_content=p.read_text(encoding="utf-8"),
            metadata={"source": p.name, "filetype": "text"},
        )
    ],
}


def load_file(path: str | Path) -> List[Document]:
    """Load any supported file type. Returns [] if extension unsupported."""
    path = Path(path)
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        return []
    return loader(path)


def load_directory(directory: str | Path) -> List[Document]:
    """Recursively load all supported files under a directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    docs: List[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            docs.extend(load_file(path))
    return docs
