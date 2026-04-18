"""
Bulk ingestion CLI.

Usage:
    python scripts/ingest.py --src data/sample_docs
    python scripts/ingest.py --src path/to/single_file.pdf
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `src.*` importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import ingest_directory, ingest_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base.")
    parser.add_argument(
        "--src",
        required=True,
        help="Path to a file or directory containing documents (PDF, Markdown, CSV, TXT).",
    )
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"ERROR: path does not exist: {src}", file=sys.stderr)
        return 1

    t0 = time.time()
    print(f"Ingesting from {src}...")
    result = ingest_file(src) if src.is_file() else ingest_directory(src)
    elapsed = time.time() - t0

    print(f"Ingested {result['chunks']} chunks from {result['files']} file(s) in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
