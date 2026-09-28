"""
Streamlit chat UI for the RAG agent.

    streamlit run app/streamlit_app.py

Features:
- Chat with multi-turn memory (persists across reruns via st.session_state).
- Sidebar: upload documents (indexed through the MCP ingest_document tool),
  list indexed docs, clear the conversation.
- Citations render as chips under each answer, plus the MCP tools the agent called.

Public demo (DEMO_MODE=true): per-session and daily question limits, upload
size and count limits, uploads expire after 30 minutes, and API keys come
from Streamlit secrets on the server. See app/demo_limits.py.
"""

from __future__ import annotations

import html
import os
import re
import sys
import threading
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _secrets_to_env() -> None:
    """Hosted deploys keep keys in Streamlit secrets (server side, never in the repo).

    Copy them into the environment before `config` is imported so the agent
    and the MCP server subprocess both see them.
    """
    try:
        items = dict(st.secrets)
    except Exception:  # no secrets file when running locally with .env
        return
    for key, value in items.items():
        if isinstance(value, (str, int, float, bool)) and key not in os.environ:
            os.environ[key] = str(value)


_secrets_to_env()

from app.demo_limits import DailyCounter, DemoLimits, SessionUsage, check_question, check_upload  # noqa: E402
from config import AGENT_MODEL, UPLOADS_SUBDIR, VECTOR_BACKEND, docs_dir  # noqa: E402
from src.agent.mcp_client import get_toolbox, list_sources  # noqa: E402
from src.pipeline import Session  # noqa: E402

DEMO_MODE = os.getenv("DEMO_MODE", "").lower() in {"1", "true", "yes"}
LIMITS = DemoLimits()
REPO_URL = "https://github.com/Majody21/rag-agent-system"
PROJECT_URL = "https://abdultaboo.netlify.app/rag-agent/"
EXAMPLE_QUESTIONS = [
    "How do I reset my password if I lost my MFA device?",
    "What is the hotel nightly limit for business travel?",
    "What documents do you have access to?",
]


# ─── Page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Knowledge Agent", page_icon="🧠", layout="wide")

st.markdown(
    """
<style>
.stApp { background: #0c1016; }
.block-container { padding-top: 2rem; max-width: 900px; }
h1, h2, h3 { color: #dddad2; letter-spacing: -0.02em; }
.stChatMessage { background: #141923; border: 1px solid rgba(255,255,255,0.06);
                 border-radius: 12px; padding: 0.5rem 1rem; }
.source-chip { display: inline-block; font-family: 'JetBrains Mono', monospace;
               font-size: 0.7rem; padding: 3px 10px; margin: 2px;
               background: rgba(0,232,170,0.12); border: 1px solid rgba(0,232,170,0.3);
               border-radius: 100px; color: #00E8AA; letter-spacing: 0.04em; }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Process-wide resources (shared by every visitor) ────────────────
@st.cache_resource
def _daily_counter() -> DailyCounter:
    return DailyCounter()


@st.cache_resource
def _upload_expiry() -> dict:
    return {"lock": threading.Lock(), "expires": {}}  # file name -> unix time


@st.cache_resource(show_spinner="Starting the MCP server and indexing the sample documents...")
def _bootstrap() -> bool:
    """Start the MCP server once per process and make sure the index is populated."""
    toolbox = get_toolbox()
    if DEMO_MODE:  # uploads left over from a previous process are removed
        for leftover in (docs_dir() / UPLOADS_SUBDIR).glob("*"):
            try:
                toolbox.call("delete_document", source=leftover.name)
            except RuntimeError:
                leftover.unlink(missing_ok=True)
    if not list_sources():
        for f in sorted((docs_dir() / "sample_docs").iterdir()):
            if f.is_file():
                toolbox.call("ingest_document", path=f"sample_docs/{f.name}")
    return True


def _expire_uploads() -> None:
    registry, now = _upload_expiry(), time.time()
    with registry["lock"]:
        for name in [n for n, t in registry["expires"].items() if t <= now]:
            try:
                get_toolbox().call("delete_document", source=name)
            except RuntimeError:
                pass
            registry["expires"].pop(name, None)


_bootstrap()
if DEMO_MODE:
    _expire_uploads()


# ─── Session state ───────────────────────────────────────────────────
st.session_state.setdefault("session", None)  # lazy init so API-key errors surface in UI
st.session_state.setdefault("messages", [])   # list of {role, content, sources?, tool_calls?}
st.session_state.setdefault("usage", SessionUsage())
st.session_state.setdefault("flash", None)


def _get_session() -> Session:
    if st.session_state.session is None:
        st.session_state.session = Session()
    return st.session_state.session


def _safe_upload_name(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)
    return f"upload_{base}" if DEMO_MODE else base


# ─── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Knowledge Agent")
    backend = "OpenSearch Serverless" if VECTOR_BACKEND == "opensearch" else "ChromaDB"
    st.caption(f"LangChain · {AGENT_MODEL} · MCP · {backend}")
    st.markdown(f"[Source code]({REPO_URL}) · [Project page]({PROJECT_URL})")

    st.divider()
    st.markdown("### 📚 Indexed Documents")
    sources = list_sources()
    if sources:
        for s in sources:
            st.markdown(f"<div class='source-chip'>{html.escape(s)}</div>", unsafe_allow_html=True)
    else:
        st.info("No documents indexed yet. Upload some below or run the ingest script.")

    st.divider()
    st.markdown("### ⬆️ Upload Documents")
    if DEMO_MODE:
        st.caption(
            f"Public demo: uploads are visible to other visitors and deleted after "
            f"{max(1, LIMITS.upload_ttl_seconds // 60)} min. Upload only public, non-sensitive files "
            f"(max {LIMITS.max_upload_bytes // 1000} KB)."
        )
    uploaded = st.file_uploader(
        "Drop PDF, Markdown, or CSV files",
        type=["pdf", "md", "markdown", "csv", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded and st.button("Ingest uploaded files", type="primary"):
        upload_dir = docs_dir() / UPLOADS_SUBDIR
        upload_dir.mkdir(parents=True, exist_ok=True)
        reports = []
        with st.spinner("Indexing through MCP ingest_document..."):
            for uf in uploaded:
                name = _safe_upload_name(uf.name)
                data = uf.getvalue()
                refusal = check_upload(len(data), st.session_state.usage, LIMITS) if DEMO_MODE else None
                if refusal is None and name in sources:
                    refusal = f"A document named {name} is already indexed."
                if refusal:
                    reports.append(f"{uf.name}: {refusal}")
                    continue
                (upload_dir / name).write_bytes(data)
                try:
                    reports.append(get_toolbox().call("ingest_document", path=f"{UPLOADS_SUBDIR}/{name}"))
                except RuntimeError as e:
                    reports.append(f"{uf.name}: could not be indexed ({e})")
                    continue
                if DEMO_MODE:
                    registry = _upload_expiry()
                    with registry["lock"]:
                        registry["expires"][name] = time.time() + LIMITS.upload_ttl_seconds
        st.session_state.flash = "\n\n".join(reports)
        st.rerun()
    if st.session_state.flash:
        st.info(st.session_state.flash)
        st.session_state.flash = None

    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.session is not None:
            st.session_state.session.reset()
        st.rerun()


# ─── Main chat area ──────────────────────────────────────────────────
st.markdown("# AI Agent & RAG")
st.caption(
    "Ask questions about the indexed documents. The agent retrieves passages through MCP "
    "tools and cites every claim."
)
if DEMO_MODE:
    st.info(
        f"Public demo on fictional sample documents for a made-up company (Acme). "
        f"Limit: {LIMITS.questions_per_session} questions per session."
    )


def _render_meta(sources: list[dict], tool_calls: list[str]) -> None:
    """Citation chips (one per distinct passage location) and the MCP tools used."""
    chips, seen = "", set()
    for src in sources:
        tag = src.get("source", "?")
        if src.get("page"):
            tag += f" · p.{src['page']}"
        elif src.get("section"):
            tag += f" · {src['section']}"
        if tag in seen:
            continue
        seen.add(tag)
        # File names come from uploads, so escape before rendering as HTML
        chips += f"<span class='source-chip'>{html.escape(tag)}</span>"
    if chips:
        st.markdown(f"<div style='margin-top:8px'>{chips}</div>", unsafe_allow_html=True)
    if tool_calls:
        st.caption("MCP tools called: " + ", ".join(tool_calls))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        _render_meta(msg.get("sources", []), msg.get("tool_calls", []))

prompt = None
if not st.session_state.messages:
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(example, use_container_width=True):
            prompt = example
prompt = st.chat_input("Ask about company policies, procedures, or documents…") or prompt

if prompt:
    refusal = (
        check_question(prompt, st.session_state.usage, _daily_counter(), LIMITS) if DEMO_MODE else None
    )
    if refusal:
        st.warning(refusal)
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_thinking…_")
        try:
            result = _get_session().ask(prompt)
        except Exception as e:
            placeholder.error(f"Error: {e}")
            st.stop()

        placeholder.markdown(result["answer"])
        _render_meta(result["sources"], result["tool_calls"])
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "tool_calls": result["tool_calls"],
            }
        )
