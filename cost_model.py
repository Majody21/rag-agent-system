"""
Monthly cost model for the RAG agent across three usage tiers.

    python cost_model.py            # prints markdown tables

Every price below was read from the official pricing page on the date in
PRICES_CHECKED and links to its source. Token counts per question were
measured on the live agent with scripts/measure_tokens.py, not estimated.
Anything that is an assumption is labelled as one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PRICES_CHECKED = "2026-09-28"
HOURS_PER_MONTH = 730
DAYS_PER_MONTH = 30

# ─── Prices (USD, us-east-1 where regional) ──────────────────────────
SOURCES = {
    "claude": "https://platform.claude.com/docs/en/about-claude/pricing",
    "gemini_embed": "https://developers.googleblog.com/gemini-embedding-available-gemini-api/",
    "gemini_pricing": "https://ai.google.dev/gemini-api/docs/pricing",
    "opensearch": "https://aws.amazon.com/opensearch-service/pricing/",
    "fargate": "https://aws.amazon.com/fargate/pricing/",
    "alb": "https://aws.amazon.com/elasticloadbalancing/pricing/",
    "s3": "https://aws.amazon.com/s3/pricing/",
    "lambda": "https://aws.amazon.com/lambda/pricing/",
}

# $ per million tokens: (input, output, cache read)
CLAUDE = {
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-sonnet-5": (2.00, 10.00, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
}
EMBED_PER_M = 0.15            # gemini-embedding-001, paid tier (GA announcement)
EMBED2_PER_M = 0.20           # gemini-embedding-2, current pricing page (for reference)
OCU_HOUR = 0.24               # OpenSearch Serverless, indexing or search OCU
OS_STORAGE_GB_MONTH = 0.02
FARGATE_VCPU_HOUR = 0.0404784  # Linux/x86
FARGATE_GB_HOUR = 0.004446
ALB_HOUR = 0.0225
ALB_LCU_HOUR = 0.008
S3_GB_MONTH = 0.023
S3_PUT_PER_1K = 0.005
LAMBDA_PER_M_REQ = 0.20
LAMBDA_GB_SECOND = 0.0000166667

# ─── Measured on the live agent (scripts/measure_tokens.py, 2026-09-28) ─
# 5 questions on claude-opus-5: single facts, a follow-up, a listing
# question, and an out-of-scope question. Includes system prompt, tool
# definitions, retrieved chunks, history, and thinking tokens (billed as output).
MEASURED_INPUT_TOKENS = 5_961
MEASURED_OUTPUT_TOKENS = 730
MEASURED_SEARCHES_PER_QUESTION = 1.6
QUERY_EMBED_TOKENS = 30       # assumption: a rewritten search query


@dataclass
class Tier:
    name: str
    queries_per_day: int
    docs_per_month: int          # assumption: new or changed documents ingested
    corpus_chunks: int           # assumption: chunks stored in the index
    host: str                    # "streamlit_cloud" or "fargate"
    fargate_tasks: int
    fargate_vcpu: float
    fargate_gb: float
    alb_lcus: float              # assumption: average load balancer capacity units
    ocus_classic: float          # OpenSearch Classic OCUs (min floor is 2 with HA)
    nextgen_warm_hours_per_day: float  # assumption for scale-to-zero NextGen
    nextgen_ocus_when_warm: float      # assumption: AWS does not publish this floor


TIERS = [
    Tier("Demo", 50, 20, 2_000, "streamlit_cloud", 0, 0, 0, 0, 1.0, 4, 1.0),
    Tier("Small team", 1_000, 500, 100_000, "fargate", 1, 1, 2, 1, 2.0, 12, 2.0),
    Tier("Department", 10_000, 5_000, 1_000_000, "fargate", 2, 2, 4, 2, 4.0, 24, 4.0),
]

PAGES_PER_DOC = 10            # assumption
TOKENS_PER_PAGE = 600         # assumption
EMBED_DIMS = 3072             # gemini-embedding-001 default
CHUNK_TEXT_BYTES = 1_500      # ~1000 chars of text + metadata
DOC_MB = 1.0                  # assumption: average raw file size in S3
LAMBDA_SECONDS_PER_DOC = 30   # assumption: ingest Lambda, 1 GB memory


def claude_monthly(model: str, queries: int, cached_fraction: float = 0.0) -> float:
    inp, out, cache_read = CLAUDE[model]
    tokens_in = queries * MEASURED_INPUT_TOKENS
    uncached = tokens_in * (1 - cached_fraction)
    return (uncached * inp + tokens_in * cached_fraction * cache_read + queries * MEASURED_OUTPUT_TOKENS * out) / 1e6


def line_items(t: Tier, model: str = "claude-opus-5") -> dict[str, float]:
    q_month = t.queries_per_day * DAYS_PER_MONTH
    items: dict[str, float] = {}
    items["Claude tokens"] = claude_monthly(model, q_month)

    ingest_tokens = t.docs_per_month * PAGES_PER_DOC * TOKENS_PER_PAGE
    query_tokens = q_month * MEASURED_SEARCHES_PER_QUESTION * QUERY_EMBED_TOKENS
    items["Embeddings"] = (ingest_tokens + query_tokens) * EMBED_PER_M / 1e6

    if t.host == "streamlit_cloud":
        items["Vector store"] = 0.0  # Chroma on disk inside the app
        items["Hosting"] = 0.0       # Streamlit Community Cloud free tier
        items["S3 + Lambda ingest"] = 0.0
    else:
        index_gb = t.corpus_chunks * (EMBED_DIMS * 4 + CHUNK_TEXT_BYTES) / 1e9
        items["Vector store"] = t.ocus_classic * OCU_HOUR * HOURS_PER_MONTH + index_gb * OS_STORAGE_GB_MONTH
        per_task = t.fargate_vcpu * FARGATE_VCPU_HOUR + t.fargate_gb * FARGATE_GB_HOUR
        items["Hosting"] = (
            t.fargate_tasks * per_task * HOURS_PER_MONTH
            + (ALB_HOUR + t.alb_lcus * ALB_LCU_HOUR) * HOURS_PER_MONTH
        )
        s3 = t.docs_per_month * DOC_MB / 1024 * 12 * S3_GB_MONTH + t.docs_per_month * S3_PUT_PER_1K / 1000
        lam = t.docs_per_month * (LAMBDA_PER_M_REQ / 1e6 + LAMBDA_SECONDS_PER_DOC * 1.0 * LAMBDA_GB_SECOND)
        items["S3 + Lambda ingest"] = s3 + lam  # before the Lambda free tier
    return items


def vector_store_options(t: Tier) -> dict[str, float]:
    index_gb = t.corpus_chunks * (EMBED_DIMS * 4 + CHUNK_TEXT_BYTES) / 1e9
    storage = index_gb * OS_STORAGE_GB_MONTH
    return {
        "Chroma (self-hosted)": 0.0,
        "OpenSearch Classic, no standby (1 OCU floor)": max(1.0, t.ocus_classic / 2) * OCU_HOUR * HOURS_PER_MONTH + storage,
        "OpenSearch Classic, standby replicas (2 OCU floor)": max(2.0, t.ocus_classic) * OCU_HOUR * HOURS_PER_MONTH + storage,
        "OpenSearch NextGen (scale to zero)": t.nextgen_ocus_when_warm * OCU_HOUR * t.nextgen_warm_hours_per_day * DAYS_PER_MONTH + storage,
    }


LEVERS = {
    "Demo": [
        "Keep the demo on Chroma: an always-on OpenSearch Classic collection would cost "
        "about $175/month (1 OCU dev-test floor) to serve $70 of Claude traffic.",
        "Cap spend with a per-session question limit and a monthly spend limit in the Claude Console.",
    ],
    "Small team": [
        "Switch the agent to claude-sonnet-5: same code, one env var, about 60% less per question. "
        "Validate with scripts/evaluate.py before switching.",
        "Cache the stable prefix (system prompt + tool definitions) and trim retrieved context "
        "(k=4 to k=3, shorter chunks) to cut the ~6,000 input tokens per question.",
    ],
    "Department": [
        "Route by difficulty: claude-haiku-4-5 for lookups and listings, a larger model only for "
        "multi-document questions.",
        "Retrieve before the first model call instead of letting the agent loop: the measured "
        "average of 1.6 searches per question means extra round trips that resend the whole context.",
    ],
}


def money(x: float) -> str:
    return f"${x:,.2f}" if x < 100 else f"${x:,.0f}"


def main() -> None:
    print(f"_Prices checked {PRICES_CHECKED}. Default agent model: claude-opus-5._\n")

    print("### Assumptions\n")
    print("| | " + " | ".join(t.name for t in TIERS) + " |")
    print("|---|" + "---:|" * len(TIERS))
    rows = [
        ("Questions per day", [f"{t.queries_per_day:,}" for t in TIERS]),
        ("Questions per month", [f"{t.queries_per_day * DAYS_PER_MONTH:,}" for t in TIERS]),
        ("Claude input tokens per question (measured)", [f"{MEASURED_INPUT_TOKENS:,}"] * 3),
        ("Claude output tokens per question (measured)", [f"{MEASURED_OUTPUT_TOKENS:,}"] * 3),
        ("Chunks retrieved per search / searches per question", ["4 / 1.6"] * 3),
        ("Documents ingested per month", [f"{t.docs_per_month:,}" for t in TIERS]),
        ("Chunks in the index", [f"{t.corpus_chunks:,}" for t in TIERS]),
        ("Hosting", ["Streamlit Community Cloud" if t.host == "streamlit_cloud"
                     else f"{t.fargate_tasks} x Fargate {t.fargate_vcpu:g} vCPU / {t.fargate_gb:g} GB + ALB" for t in TIERS]),
        ("Vector store", ["Chroma in the app"] + [f"OpenSearch Serverless, {t.ocus_classic:g} OCU" for t in TIERS[1:]]),
    ]
    for label, vals in rows:
        print(f"| {label} | " + " | ".join(vals) + " |")

    print("\n### Monthly cost by tier\n")
    all_items = [line_items(t) for t in TIERS]
    keys = list(all_items[0])
    print("| Line item | " + " | ".join(t.name for t in TIERS) + " |")
    print("|---|" + "---:|" * len(TIERS))
    for k in keys:
        print(f"| {k} | " + " | ".join(money(i[k]) for i in all_items) + " |")
    print("| **Total per month** | " + " | ".join(f"**{money(sum(i.values()))}**" for i in all_items) + " |")
    print("| Cost per question | " + " | ".join(
        f"${sum(i.values()) / (t.queries_per_day * DAYS_PER_MONTH):.3f}" for i, t in zip(all_items, TIERS)) + " |")

    print("\n### Claude cost by model (the biggest lever)\n")
    print("| Model | Per question | " + " | ".join(t.name for t in TIERS) + " |")
    print("|---|---:|" + "---:|" * len(TIERS))
    for m in CLAUDE:
        per_q = claude_monthly(m, 1)
        print(f"| {m} | ${per_q:.4f} | " + " | ".join(
            money(claude_monthly(m, t.queries_per_day * DAYS_PER_MONTH)) for t in TIERS) + " |")

    print("\n### Vector store options\n")
    opts = [vector_store_options(t) for t in TIERS]
    print("| Option | " + " | ".join(t.name for t in TIERS) + " |")
    print("|---|" + "---:|" * len(TIERS))
    for k in opts[0]:
        print(f"| {k} | " + " | ".join(money(o[k]) for o in opts) + " |")

    print("\n### Biggest cost driver and how to cut it\n")
    for t, items in zip(TIERS, all_items):
        driver = max(items, key=items.get)
        share = items[driver] / sum(items.values()) * 100
        if t.name == "Demo":
            alt = vector_store_options(t)["OpenSearch Classic, no standby (1 OCU floor)"]
            print(f"- **{t.name}**: {driver} ({share:.0f}% of {money(sum(items.values()))}). "
                  f"If the demo ran on OpenSearch Classic instead of Chroma, the OCU floor "
                  f"({money(alt)}) would be the biggest line.")
        else:
            print(f"- **{t.name}**: {driver} ({share:.0f}% of {money(sum(items.values()))}).")
        for lever in LEVERS[t.name]:
            print(f"  - {lever}")

    print("\n### Notes\n")
    print("- OpenSearch Classic collections bill at least 2 OCUs with standby replicas, or 1 OCU "
          "(0.5 indexing + 0.5 search) in dev-test mode, even when idle. NextGen collections scale "
          "to zero after 10 minutes idle; AWS does not publish the OCU count while warm, so the "
          "NextGen row uses the OCU assumption in `TIERS`.")
    print("- OCU counts for the team and department tiers are sizing assumptions to confirm with a "
          "load test. At 3,072 dimensions a million chunks is about 14 GB of vectors; cutting to "
          "768 dimensions (supported by gemini-embedding-001) shrinks that 4x.")
    print("- Hosting, S3, and Lambda rows are the planned AWS deployment, not something running today. "
          "Lambda is shown before its monthly free tier (1M requests, 400,000 GB-seconds).")
    print(f"- gemini-embedding-001 is priced at ${EMBED_PER_M:.2f}/M tokens per Google's GA announcement; "
          f"the current pricing page lists only gemini-embedding-2 (${EMBED2_PER_M:.2f}/M). Either way "
          "embeddings stay under 1% of the total.")
    print("- Claude 4.7 and later models use a tokenizer that produces about 30% more tokens for the "
          "same text than Haiku 4.5, so the Haiku row slightly overstates its cost.")

    print("\n### Sources\n")
    for name, url in SOURCES.items():
        print(f"- {name}: {url}")


if __name__ == "__main__":
    main()
