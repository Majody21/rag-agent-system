"""
Interactive CLI for the RAG agent.

    python -m app.cli

Commands:
    /reset   clear conversation memory
    /docs    list indexed documents
    /quit    exit
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import Session  # noqa: E402
from src.agent.mcp_client import get_toolbox, list_sources  # noqa: E402



# ANSI colors (safe on modern Windows terminals and *nix)
GREEN = "\033[32m"
BLUE = "\033[34m"
GREY = "\033[90m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _banner() -> None:
    print(f"{BOLD}{CYAN}┌────────────────────────────────────────────┐{RESET}")
    print(f"{BOLD}{CYAN}│  AI Agent & RAG: Enterprise Knowledge Q&A  │{RESET}")
    print(f"{BOLD}{CYAN}└────────────────────────────────────────────┘{RESET}")
    sources = list_sources()
    if sources:
        print(f"{GREY}Indexed: {', '.join(sources)}{RESET}")
    else:
        print(f"{GREY}No documents indexed. Run: python scripts/ingest.py --src data/sample_docs{RESET}")
    print(f"{GREY}Commands: /reset  /docs  /quit{RESET}\n")


def main() -> int:
    session = Session()
    _banner()

    while True:
        try:
            q = input(f"{BOLD}you ▸ {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not q:
            continue
        if q in {"/quit", "/exit", ":q"}:
            return 0
        if q == "/reset":
            session.reset()
            print(f"{GREY}(memory cleared){RESET}\n")
            continue
        if q == "/docs":
            srcs = list_sources()
            print(f"{GREY}" + ("\n".join(f"  - {s}" for s in srcs) or "  (none)") + f"{RESET}\n")
            continue

        try:
            result = session.ask(q)
        except Exception as e:
            print(f"\033[31merror: {e}{RESET}\n")
            continue

        print(f"\n{BOLD}{GREEN}agent ▸{RESET} {result['answer']}\n")
        if result["sources"]:
            print(f"{GREY}sources:{RESET}")
            seen: set[tuple] = set()
            for s in result["sources"]:
                key = (s.get("source"), s.get("page"), s.get("chunk"))
                if key in seen:
                    continue
                seen.add(key)
                tag = s.get("source", "?")
                if s.get("page"):
                    tag += f" · p.{s['page']}"
                if s.get("section"):
                    tag += f" · {s['section']}"
                print(f"{GREY}  · {BLUE}{tag}{RESET}")
            print()


if __name__ == "__main__":
    raise SystemExit(main())
