"""
Measure real Claude token usage per question, for the cost model.

Runs a fixed set of questions against the live agent (current .env
settings) and prints input/output tokens per question and the averages.
The averages feed MEASURED_* in cost_model.py.

    python scripts/ingest.py --src data/sample_docs
    python scripts/measure_tokens.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import Session  # noqa: E402

# (question, start a fresh session?) -- a mix of single facts, a follow-up,
# a listing question, and an out-of-scope question the agent must refuse.
QUESTIONS = [
    ("How do I reset my password if I still have access to my MFA device?", True),
    ("What if I've also lost my MFA device?", False),
    ("How long do I have to submit an expense for reimbursement?", True),
    ("What documents do you have access to?", True),
    ("What is the company's parental leave policy?", True),
]


def main() -> int:
    session = Session()
    inputs, outputs = [], []
    for question, fresh in QUESTIONS:
        if fresh:
            session.reset()
        t0 = time.time()
        result = session.ask(question)
        usage = next(iter(result["usage"].values()))
        inputs.append(usage["input_tokens"])
        outputs.append(usage["output_tokens"])
        print(
            f"{usage['input_tokens']:>6} in {usage['output_tokens']:>5} out "
            f"{time.time() - t0:5.1f}s  tools={result['tool_calls']}  {question[:50]}"
        )
    print(f"\nmean input tokens : {statistics.mean(inputs):,.0f}")
    print(f"mean output tokens: {statistics.mean(outputs):,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
