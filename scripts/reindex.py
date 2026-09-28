"""
Re-indexing.

Two modes:

    # Full rebuild: wipe the store and re-ingest everything
    python scripts/reindex.py --src data/sample_docs --full --yes

    # Automated: watch a folder and re-index only what changed
    python scripts/reindex.py --src data/sample_docs --watch --interval 30

Watch mode hashes every supported file each interval. New or edited files
are re-ingested (old chunks for that file are replaced), deleted files are
removed from the index. A manifest of hashes survives restarts, so a
restart does not re-embed unchanged files. On AWS the same step would be an
S3 event notification triggering a Lambda that calls `ingest_file`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT  # noqa: E402
from src.pipeline import ingest_directory, sync_directory  # noqa: E402
from src.vectorstore import reset_store  # noqa: E402

MANIFEST = PROJECT_ROOT / ".index_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-index the knowledge base.")
    parser.add_argument("--src", required=True, help="Directory of documents.")
    parser.add_argument("--full", action="store_true", help="Wipe the store and rebuild from scratch.")
    parser.add_argument("--watch", action="store_true", help="Keep running and re-index on change.")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between checks in watch mode.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt for --full.")
    args = parser.parse_args()

    if args.full:
        if not args.yes:
            confirm = input("This will DELETE the existing vector store. Continue? [y/N] ")
            if confirm.strip().lower() not in {"y", "yes"}:
                print("Aborted.")
                return 1
        print("Wiping existing store...")
        reset_store()
        MANIFEST.unlink(missing_ok=True)
        t0 = time.time()
        result = ingest_directory(args.src)
        print(f"Done. {result['chunks']} chunks from {result['files']} files in {time.time() - t0:.1f}s")

    while True:
        report = sync_directory(args.src, MANIFEST)
        changed = {k: v for k, v in report.items() if v}
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] " + (f"re-indexed {changed}" if changed else "no changes"), flush=True)
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
