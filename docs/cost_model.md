_Prices checked 2026-09-28. Default agent model: claude-opus-5._

### Assumptions

| | Demo | Small team | Department |
|---|---:|---:|---:|
| Questions per day | 50 | 1,000 | 10,000 |
| Questions per month | 1,500 | 30,000 | 300,000 |
| Claude input tokens per question (measured) | 5,961 | 5,961 | 5,961 |
| Claude output tokens per question (measured) | 730 | 730 | 730 |
| Chunks retrieved per search / searches per question | 4 / 1.6 | 4 / 1.6 | 4 / 1.6 |
| Documents ingested per month | 20 | 500 | 5,000 |
| Chunks in the index | 2,000 | 100,000 | 1,000,000 |
| Hosting | Streamlit Community Cloud | 1 x Fargate 1 vCPU / 2 GB + ALB | 2 x Fargate 2 vCPU / 4 GB + ALB |
| Vector store | Chroma in the app | OpenSearch Serverless, 2 OCU | OpenSearch Serverless, 4 OCU |

### Monthly cost by tier

| Line item | Demo | Small team | Department |
|---|---:|---:|---:|
| Claude tokens | $72.08 | $1,442 | $14,416 |
| Embeddings | $0.03 | $0.67 | $6.66 |
| Vector store | $0.00 | $350 | $701 |
| Hosting | $0.00 | $58.31 | $172 |
| S3 + Lambda ingest | $0.00 | $0.39 | $3.87 |
| **Total per month** | **$72.11** | **$1,851** | **$15,300** |
| Cost per question | $0.048 | $0.062 | $0.051 |

### Claude cost by model (the biggest lever)

| Model | Per question | Demo | Small team | Department |
|---|---:|---:|---:|---:|
| claude-opus-5 | $0.0481 | $72.08 | $1,442 | $14,416 |
| claude-sonnet-5 | $0.0192 | $28.83 | $577 | $5,767 |
| claude-haiku-4-5 | $0.0096 | $14.42 | $288 | $2,883 |

### Vector store options

| Option | Demo | Small team | Department |
|---|---:|---:|---:|
| Chroma (self-hosted) | $0.00 | $0.00 | $0.00 |
| OpenSearch Classic, no standby (1 OCU floor) | $175 | $175 | $351 |
| OpenSearch Classic, standby replicas (2 OCU floor) | $350 | $350 | $701 |
| OpenSearch NextGen (scale to zero) | $28.80 | $173 | $691 |

### Biggest cost driver and how to cut it

- **Demo**: Claude tokens (100% of $72.11). If the demo ran on OpenSearch Classic instead of Chroma, the OCU floor ($175) would be the biggest line.
  - Keep the demo on Chroma: an always-on OpenSearch Classic collection would cost about $175/month (1 OCU dev-test floor) to serve $70 of Claude traffic.
  - Cap spend with a per-session question limit and a monthly spend limit in the Claude Console.
- **Small team**: Claude tokens (78% of $1,851).
  - Switch the agent to claude-sonnet-5: same code, one env var, about 60% less per question. Validate with scripts/evaluate.py before switching.
  - Cache the stable prefix (system prompt + tool definitions) and trim retrieved context (k=4 to k=3, shorter chunks) to cut the ~6,000 input tokens per question.
- **Department**: Claude tokens (94% of $15,300).
  - Route by difficulty: claude-haiku-4-5 for lookups and listings, a larger model only for multi-document questions.
  - Retrieve before the first model call instead of letting the agent loop: the measured average of 1.6 searches per question means extra round trips that resend the whole context.

### Notes

- OpenSearch Classic collections bill at least 2 OCUs with standby replicas, or 1 OCU (0.5 indexing + 0.5 search) in dev-test mode, even when idle. NextGen collections scale to zero after 10 minutes idle; AWS does not publish the OCU count while warm, so the NextGen row uses the OCU assumption in `TIERS`.
- OCU counts for the team and department tiers are sizing assumptions to confirm with a load test. At 3,072 dimensions a million chunks is about 14 GB of vectors; cutting to 768 dimensions (supported by gemini-embedding-001) shrinks that 4x.
- Hosting, S3, and Lambda rows are the planned AWS deployment, not something running today. Lambda is shown before its monthly free tier (1M requests, 400,000 GB-seconds).
- gemini-embedding-001 is priced at $0.15/M tokens per Google's GA announcement; the current pricing page lists only gemini-embedding-2 ($0.20/M). Either way embeddings stay under 1% of the total.
- Claude 4.7 and later models use a tokenizer that produces about 30% more tokens for the same text than Haiku 4.5, so the Haiku row slightly overstates its cost.

### Sources

- claude: https://platform.claude.com/docs/en/about-claude/pricing
- gemini_embed: https://developers.googleblog.com/gemini-embedding-available-gemini-api/
- gemini_pricing: https://ai.google.dev/gemini-api/docs/pricing
- opensearch: https://aws.amazon.com/opensearch-service/pricing/
- fargate: https://aws.amazon.com/fargate/pricing/
- alb: https://aws.amazon.com/elasticloadbalancing/pricing/
- s3: https://aws.amazon.com/s3/pricing/
- lambda: https://aws.amazon.com/lambda/pricing/
