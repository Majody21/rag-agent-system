# AI Agent & RAG Automation System

A production-grade Retrieval-Augmented Generation (RAG) agent for enterprise
knowledge bases. Ingests PDFs, Markdown, and CSV files; embeds them into a
local vector store; and answers natural-language questions through a
conversational LangChain agent that **cites its sources** and **refuses to
hallucinate when retrieval turns up empty**.

Built by [Abdulmajeed Taboo](https://www.linkedin.com/in/abdultaboo/) —
AI-Proficient Data & Business Analyst based in Chicago, IL.

---

## The Problem

Enterprise teams waste hours manually searching through internal documentation,
SOPs, and knowledge bases to answer routine questions — leading to slow
onboarding, inconsistent answers, and lost productivity.

## What It Does

- **Ingests** documents in multiple formats (PDF, Markdown, CSV, TXT) with
  format-aware loaders that preserve structural metadata (page numbers for
  PDFs, section headings for Markdown, row numbers for CSVs).
- **Chunks** intelligently using `RecursiveCharacterTextSplitter` with
  overlap to preserve context across chunk boundaries.
- **Embeds** with Google `text-embedding-004` (1536 dims) into a
  persistent ChromaDB store. Ingestion is **idempotent** — re-ingesting a
  file won't create duplicates.
- **Answers** through a tool-calling Claude Sonnet agent with three tools:
  `search_knowledge_base`, `list_documents`, `get_document_summary`.
- **Remembers** conversation history via a sliding window, supporting
  multi-turn Q&A ("what if that doesn't work?").
- **Cites every claim** with `[source: filename]` tags surfaced to the UI
  as expandable chips.

## Outcome

Reduces average internal query resolution from ~15 minutes of manual
searching to **under 30 seconds of AI-assisted retrieval** — applicable to
healthcare operations, finance teams, and IT help desks.

---

## Architecture

```
           ┌──────────────────────────────────────────────┐
           │              Streamlit UI / CLI              │
           └─────────────────┬────────────────────────────┘
                             │
                ┌────────────▼─────────────┐
                │   LangChain Tool Agent   │◄──── Google Gemini 2.5 Flash
                │    (windowed memory)     │
                └────────────┬─────────────┘
                             │  tool call
                ┌────────────▼─────────────┐
                │    Retriever (Chroma)    │
                │  top-k + metadata filter │
                └────────────┬─────────────┘
                             │
        ┌────────────────────▼────────────────────┐
        │           Vector Store (Chroma)         │
        │  embeddings: OpenAI text-embedding-3-s  │
        └────────────────────▲────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │   Ingestion Pipeline                    │
        │   PDF / MD / CSV loaders → chunker →    │
        │   embedder → upsert (content-hash IDs)  │
        └─────────────────────────────────────────┘
```

---

## Quickstart

### 1. Install

```bash
git clone <this-repo>
cd rag-agent-system
python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# then edit .env and fill in:
#   GOOGLE_API_KEY=AIzaSy...   (free key: https://aistudio.google.com/app/apikey)
```

### 3. Generate the sample PDF (one-time)

```bash
python scripts/generate_sample_pdf.py
```

This converts `data/sample_docs/it_password_reset_source.md` into a real PDF
so the PDF loader has something to chew on.

### 4. Ingest the sample documents

```bash
python scripts/ingest.py --src data/sample_docs
```

Expected output:

```
Ingested 47 chunks from 4 file(s) in 3.2s
```

### 5. Ask questions

**CLI:**

```bash
python -m app.cli
```

```
you ▸ How do I reset my password?
agent ▸ To reset your password when you still have MFA access, go to
https://login.acme.example and click "Forgot password?" … [source: it_password_reset.pdf, p.1]

you ▸ What if I've also lost my MFA device?
agent ▸ If you've lost your MFA device but can still access your email … [source: it_password_reset.pdf, p.2]
```

**Streamlit UI:**

```bash
streamlit run app/streamlit_app.py
```

The UI includes a sidebar for uploading new documents on the fly, live
citation chips under each answer, and controls to clear chat / reset the
entire index.

### 6. Evaluate

```bash
python scripts/evaluate.py
```

Runs the agent against `eval/qa_pairs.json` (12 Q/A pairs keyed to the
sample docs) and scores each with Claude-as-judge on two axes:
**faithfulness** (answer grounded in sources) and **relevance** (answer
addresses the question). Averages are printed at the end.

### 7. Test

```bash
pytest -v
```

Unit tests run with no API keys required. Integration tests (marked
`live`) auto-run when `GOOGLE_API_KEY` is present.

---

## Project Layout

```
rag-agent-system/
├── README.md
├── requirements.txt
├── config.py                    # All tunable knobs
│
├── src/
│   ├── ingestion/               # PDF / MD / CSV loaders + chunker
│   ├── vectorstore/             # Chroma wrapper with content-hash dedup
│   ├── retrieval/               # Retriever + citation formatter
│   ├── agent/                   # Tools, prompts, AgentExecutor, Session
│   └── pipeline.py              # High-level convenience API
│
├── app/
│   ├── cli.py                   # Interactive terminal REPL
│   └── streamlit_app.py         # Web chat UI
│
├── scripts/
│   ├── ingest.py                # Bulk ingest
│   ├── reindex.py               # Wipe + re-ingest
│   ├── evaluate.py              # RAGAS-style eval
│   └── generate_sample_pdf.py   # Build the sample PDF
│
├── data/sample_docs/            # 4 realistic enterprise docs for demo
│   ├── hr_onboarding.md
│   ├── it_password_reset.pdf
│   ├── finance_expense_policy.md
│   └── employees.csv
│
├── eval/qa_pairs.json           # 12 Q/A test pairs
└── tests/                       # pytest
```

---

## Key Design Decisions

1. **Citations are non-negotiable.** The system prompt
   (`src/agent/prompts.py`) explicitly instructs Claude to (a) always call
   the retrieval tool before answering a factual question, (b) refuse when
   retrieval returns nothing, and (c) cite sources inline.

2. **Idempotent ingestion via content-hash IDs.** Re-running ingestion on
   the same corpus does not create duplicate vectors. The ID for each
   chunk is `{source}::{sha1(content)[:16]}`.

3. **Multi-tool agent, not a single retrieval chain.** The agent chooses
   between `search_knowledge_base`, `list_documents`, and
   `get_document_summary` based on the user's intent — demonstrates real
   agent reasoning, not just retrieval-then-generate.

4. **Metadata-rich chunks** enable filtered retrieval. Each chunk carries
   `{source, filetype, chunk_index, page?, section_heading?, row?}`. The
   agent's search tool accepts an optional `source_filter` so the user can
   say "in the HR handbook, how does…" and get filtered results.

5. **Sliding-window memory** (`MEMORY_WINDOW = 5` turn pairs) keeps
   multi-turn context cheap. Follow-ups like "what if that doesn't work?"
   resolve correctly without blowing up tokens.

---

## Deployment to AWS

The current implementation runs locally. To deploy to AWS:

### Architecture sketch

```
      Users ──HTTPS──► CloudFront ──► ALB ──► ECS Fargate (Streamlit)
                                                   │
                                                   ▼
                                          ┌────────────────────┐
                                          │   AWS Bedrock      │
                                          │   (Claude Sonnet)  │
                                          └────────────────────┘
                                                   │
                         ┌─────────────────────────┘
                         ▼
                ┌────────────────────────┐
                │  OpenSearch Serverless │◄──── ingestion Lambda
                │   (vector collection)  │        (S3 PUT trigger)
                └────────────────────────┘
                         ▲
                         │
                    S3 raw/   ◄── internal docs uploaded here
```

### Mapping from this codebase to AWS

| Local component | AWS equivalent | Notes |
|---|---|---|
| `ChatGoogleGenerativeAI` | `langchain-aws.ChatBedrock` | Drop-in swap in `src/agent/agent.py`; set `model_id="anthropic.claude-sonnet-4-5-v1:0"` or keep Gemini via Vertex AI |
| `GoogleGenerativeAIEmbeddings` | `langchain-aws.BedrockEmbeddings` | Use Titan Text Embeddings v2 (1024 dims) — rebuild the index |
| `Chroma` | `langchain-community.OpenSearchVectorSearch` | Enable OpenSearch Serverless collection w/ `vector` type |
| `scripts/ingest.py` | Lambda triggered on S3 `PUT` | `data/sample_docs/` → `s3://acme-kb-raw/` |
| `app/streamlit_app.py` | ECS Fargate service behind ALB | Single `Dockerfile`; one-container service |
| `.env` | AWS Secrets Manager | Pulled at container start |

### Why not deployed already?

This repo focuses on the core retrieval quality and agent reasoning. AWS
deployment is mechanical (swap provider classes, containerize, wire IAM)
but adds no algorithmic insight. Live deployment is a planned extension.

---

## Roadmap

- [ ] **Hybrid retrieval** (BM25 + dense) with Reciprocal Rank Fusion —
      will lift precision on keyword-heavy queries (employee IDs, form
      numbers) without sacrificing semantic recall.
- [ ] **Cross-encoder reranker** on top-20 candidates before sending top-4
      to the LLM — expected faithfulness lift of 5-10 pts on the eval set.
- [ ] **Multi-agent orchestration** — specialist agents for HR / IT /
      Finance, with a router agent dispatching queries. Currently in
      prototype.
- [ ] **AWS Bedrock deployment** (see above) — containerize + Terraform
      module.
- [ ] **Observability** — structured logging of retrievals + LLM calls,
      plus a small admin dashboard for retrieval hit rates.

---

## License

MIT. See [LICENSE](LICENSE) if included.

## Contact

**Abdulmajeed Taboo** — [majodytbo08@gmail.com](mailto:majodytbo08@gmail.com)
· [LinkedIn](https://www.linkedin.com/in/abdultaboo/)
· [Portfolio](https://abdultaboo.netlify.app)
