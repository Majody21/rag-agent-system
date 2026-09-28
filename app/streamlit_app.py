"""
Streamlit chat UI for the RAG agent.

    streamlit run app/streamlit_app.py

Features:
- Chat with multi-turn memory (persists across reruns via st.session_state).
- Sidebar: upload documents (indexed through the MCP ingest_document tool),
  list indexed docs, clear the conversation.
- Citations render as chips under each answer, plus the MCP tools the agent called.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AGENT_MODEL, UPLOADS_SUBDIR, VECTOR_BACKEND, docs_dir  # noqa: E402
from src.agent.mcp_client import get_toolbox, list_sources  # noqa: E402
from src.pipeline import Session  # noqa: E402



# ─── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Knowledge Agent",
    page_icon="🧠",
    layout="wide",
)


# ─── Minimal custom styling ──────────────────────────────────────────
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


# ─── Session state ───────────────────────────────────────────────────
if "session" not in st.session_state:
    st.session_state.session = None  # lazy init so API-key errors surface in UI
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources?}


def _get_session() -> Session:
    if st.session_state.session is None:
        st.session_state.session = Session()
    return st.session_state.session


# ─── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Knowledge Agent")
    st.caption(f"LangChain · {AGENT_MODEL} · {'OpenSearch Serverless' if VECTOR_BACKEND == 'opensearch' else 'ChromaDB'}")

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
    uploaded = st.file_uploader(
        "Drop PDF, Markdown, or CSV files",
        type=["pdf", "md", "markdown", "csv", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded and st.button("Ingest uploaded files", type="primary"):
        upload_dir = docs_dir() / UPLOADS_SUBDIR
        upload_dir.mkdir(parents=True, exist_ok=True)
        with st.spinner("Indexing through MCP ingest_document..."):
            reports = []
            for uf in uploaded:
                target = upload_dir / Path(uf.name).name
                target.write_bytes(uf.getvalue())
                reports.append(get_toolbox().call("ingest_document", path=f"{UPLOADS_SUBDIR}/{target.name}"))
        st.success("\n\n".join(reports))
        st.rerun()

    st.divider()

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.session is not None:
            st.session_state.session.reset()
        st.rerun()


# ─── Main chat area ──────────────────────────────────────────────────
st.markdown("# AI Agent & RAG")
st.caption("Ask questions about your internal documents. Answers are grounded in retrieved sources with citations.")

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


# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        _render_meta(msg.get("sources", []), msg.get("tool_calls", []))

# Input
prompt = st.chat_input("Ask about company policies, procedures, or documents…")
if prompt:
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
