"""
RAGAS-style evaluation with no heavy RAGAS dependency.

We score each Q/A pair along two axes via Claude-as-judge:
  - faithfulness : is the answer grounded in the retrieved sources?
  - relevance    : does the answer actually address the question?

Each returns a float in [0, 1]. Report per-question scores + averages.

    python scripts/evaluate.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from config import EVAL_DIR, JUDGE_MODEL  # noqa: E402
from src.agent.agent import build_llm  # noqa: E402
from src.pipeline import Session  # noqa: E402


JUDGE_SYSTEM = """You are a strict evaluator for RAG system outputs. You will
score a single dimension of quality on a 0.0 to 1.0 scale.

Respond ONLY with a JSON object of the form:
  {"score": 0.0-1.0, "reason": "one sentence"}

No prose, no markdown fences, just the JSON.
"""

FAITHFULNESS_TEMPLATE = """Evaluate FAITHFULNESS of the answer.

- 1.0 = Every factual claim in the answer is supported by the retrieved context.
- 0.5 = Partially supported; some claims unsupported but not contradicted.
- 0.0 = Answer contradicts the context or invents facts not present.

Question: {question}

Retrieved context tags (the sources the agent pulled):
{sources}

Agent answer:
{answer}

Return JSON only."""

RELEVANCE_TEMPLATE = """Evaluate RELEVANCE of the answer to the question.

- 1.0 = Directly and completely addresses the question.
- 0.5 = Addresses it partially or with unnecessary tangents.
- 0.0 = Off-topic or non-responsive (e.g. refuses without reason).

Question: {question}

Answer:
{answer}

Return JSON only."""


def _judge(prompt: str) -> float:
    resp = build_llm(JUDGE_MODEL).invoke([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=prompt)])
    text = resp.text
    # Be liberal: find the first {...} blob
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return 0.0
    try:
        return float(json.loads(match.group(0)).get("score", 0.0))
    except Exception:
        return 0.0


def main() -> int:
    qa_path = EVAL_DIR / "qa_pairs.json"
    if not qa_path.exists():
        print(f"ERROR: {qa_path} not found. Create it or run ingestion first.", file=sys.stderr)
        return 1

    qa_pairs = json.loads(qa_path.read_text(encoding="utf-8"))
    print(f"Evaluating {len(qa_pairs)} Q/A pairs against the live agent...\n")

    session = Session()
    faithfulness_scores: list[float] = []
    relevance_scores: list[float] = []

    for i, pair in enumerate(qa_pairs, start=1):
        q = pair["question"]
        session.reset()  # fresh context per eval question
        t0 = time.time()
        result = session.ask(q)
        elapsed = time.time() - t0

        source_tags = ", ".join(
            f"{s.get('source')}{':p' + str(s.get('page')) if s.get('page') else ''}"
            for s in result["sources"]
        ) or "(none)"

        faith = _judge(
            FAITHFULNESS_TEMPLATE.format(question=q, sources=source_tags, answer=result["answer"])
        )
        rel = _judge(RELEVANCE_TEMPLATE.format(question=q, answer=result["answer"]))

        faithfulness_scores.append(faith)
        relevance_scores.append(rel)

        print(f"[{i}/{len(qa_pairs)}] {q}")
        print(f"    faithfulness={faith:.2f}  relevance={rel:.2f}  ({elapsed:.1f}s)")
        print(f"    sources: {source_tags}")
        print()

    print("─" * 60)
    print(f"Average faithfulness : {mean(faithfulness_scores):.3f}")
    print(f"Average relevance    : {mean(relevance_scores):.3f}")
    print(f"N questions          : {len(qa_pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
