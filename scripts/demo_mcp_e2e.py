"""
End-to-end proof that the agent's tools are served over MCP.

    python scripts/demo_mcp_e2e.py "How do I reset my password if I lost my MFA device?"

Prints: the tools the agent discovered from the MCP server, each MCP tool
call with its arguments, the parsed citations, and Claude's cited answer.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AGENT_MODEL  # noqa: E402
from src.agent.mcp_client import SERVER_SCRIPT, get_toolbox  # noqa: E402
from src.pipeline import Session  # noqa: E402

DEFAULT_Q = "How do I reset my password if I lost my MFA device?"


def main() -> int:
    question = " ".join(sys.argv[1:]) or DEFAULT_Q
    toolbox = get_toolbox()
    print(f"MCP server  : {SERVER_SCRIPT.name} over stdio")
    print(f"Server tools: {[t.name for t in toolbox.tools]}")
    print(f"Agent tools : {[t.name for t in toolbox.agent_tools]} (read-only subset)")
    print(f"Model       : {AGENT_MODEL}\n")
    print(f"QUESTION: {question}\n")

    t0 = time.time()
    result = Session().ask(question)
    for i, call in enumerate(result["tool_trace"], 1):
        print(f"MCP tool call {i}: {call['tool']}({json.dumps(call['input'])})")
    seen = []
    for s in result["sources"]:
        tag = ", ".join(f"{k}={v}" for k, v in s.items())
        if tag not in seen:
            seen.append(tag)
    print("\nCitations returned by search_documents:")
    for tag in seen:
        print(f"  [{tag}]")
    print(f"\nANSWER ({time.time() - t0:.1f}s):\n{result['answer']}")
    usage = next(iter(result["usage"].values()), {})
    print(f"\nTokens: {usage.get('input_tokens')} in, {usage.get('output_tokens')} out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
