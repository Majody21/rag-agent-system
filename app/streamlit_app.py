"""
Streamlit chat UI for the RAG agent.

    streamlit run app/streamlit_app.py

Features:
- Chat with multi-turn memory (persists across reruns via st.session_state).
- Sidebar: upload documents for live ingestion, list indexed docs, reset memory.
- Citations render as expandable chips under each answer.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import Session, ingest_file  # noqa: E402
from src.vectorstore import list_sources, reset_store  # noqa: E402


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
    st.caption("LangChain · Claude Sonnet · ChromaDB")

    st.divider()

    st.markdown("### 📚 Indexed Documents")
    sources = list_sources()
    if sources:
        for s in sources:
            st.markdown(f"<div class='source-chip'>{s}</div>", unsafe_allow_html=True)
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
        with st.spinner("Indexing..."):
            total = 0
            for uf in uploaded:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(uf.name).suffix
                ) as tmp:
                    tmp.write(uf.read())
                    tmp_path = Path(tmp.name)
                # Rename so metadata preserves real filename
                final_path = tmp_path.with_name(uf.name)
                tmp_path.rename(final_path)
                try:
                    result = ingest_file(final_path)
                    total += result["chunks"]
                finally:
                    try:
                        final_path.unlink()
                    except OSError:
                        pass
        st.success(f"Ingested {total} chunks from {len(uploaded)} file(s).")
        st.rerun()

    st.divider()

    col1, col2 = st.columns(2)
    if col1.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.session is not None:
            st.session_state.session.reset()
        st.rerun()
    if col2.button("🗑️ Reset index", use_container_width=True):
        reset_store()
        st.session_state.messages = []
        st.session_state.session = None
        st.rerun()


# ─── Main chat area ──────────────────────────────────────────────────
st.markdown("# AI Agent & RAG")
st.caption("Ask questions about your internal documents. Answers are grounded in retrieved sources with citations.")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            chips = ""
            seen: set[tuple] = set()
            for s in msg["sources"]:
                key = (s.get("source"), s.get("page"), s.get("chunk"))
                if key in seen:
                    continue
                seen.add(key)
                tag = s.get("source", "?")
                if s.get("page"):
                    tag += f" · p.{s['page']}"
                chips += f"<span class='source-chip'>{tag}</span>"
            if chips:
                st.markdown(f"<div style='margin-top:8px'>{chips}</div>", unsafe_allow_html=True)

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
        if result["sources"]:
            chips = ""
            seen: set[tuple] = set()
            for s in result["sources"]:
                key = (s.get("source"), s.get("page"), s.get("chunk"))
                if key in seen:
                    continue
                seen.add(key)
                tag = s.get("source", "?")
                if s.get("page"):
                    tag += f" · p.{s['page']}"
                chips += f"<span class='source-chip'>{tag}</span>"
            if chips:
                st.markdown(f"<div style='margin-top:8px'>{chips}</div>", unsafe_allow_html=True)

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        )
