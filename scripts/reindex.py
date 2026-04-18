"""
Wipe the Chroma store and re-ingest from a directory.

Use this when you change the embedding model, chunk size, or otherwise
need a clean rebuild.

    python scripts/reindex.py --src data/sample_docs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import ingest_directory  # noqa: E402
from src.vectorstore import reset_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset + re-ingest the knowledge base.")
    parser.add_argument("--src", required=True, help="Directory of documents to re-ingest.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    args = parser.parse_args()

    if not args.yes:
        confirm = input("This will DELETE the existing vector store. Continue? [y/N] ")
        if confirm.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1

    print("Wiping existing store...")
    reset_store()

    t0 = time.time()
    print(f"Re-ingesting from {args.src}...")
    result = ingest_directory(args.src)
    elapsed = time.time() - t0

    print(f"Done. {result['chunks']} chunks from {result['files']} files in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
