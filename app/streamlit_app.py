"""
Streamlit chat UI for the RAG agent.

    streamlit run app/streamlit_app.py

Features:
- Chat with multi-turn memory (persists across reruns via st.session_state).
- Sidebar: upload documents (indexed through the MCP ingest_document tool),
  list indexed docs, clear the conversation.
- Citations render as chips under each answer, with an expandable trace of the
  MCP tool calls behind it. Styling lives in app/theme.css.

Public demo (DEMO_MODE=true): per-session and daily question limits, upload
size and count limits, uploads expire after 30 minutes, and API keys come
from Streamlit secrets on the server. See app/demo_limits.py.
"""

from __future__ import annotations

import html
import json
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
st.set_page_config(page_title="Knowledge Agent | RAG over MCP", page_icon=":material/travel_explore:", layout="wide")
st.html(f"<style>{(ROOT / 'app' / 'theme.css').read_text(encoding='utf-8')}</style>")

USER_AVATAR = ":material/person:"
AGENT_AVATAR = ":material/travel_explore:"


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


# ─── Small render helpers ────────────────────────────────────────────
def _ext_badge(name: str) -> str:
    ext = Path(name).suffix.lower().lstrip(".")
    return {"markdown": "MD"}.get(ext, ext.upper()[:4] or "DOC")


def _render_citations(sources: list[dict]) -> None:
    """Sources grouped by document: each file once, then the pages or sections cited.

    Built from what retrieval returned, not from the model's text.
    """
    groups: dict[str, list[str]] = {}
    for src in sources:
        name = src.get("source", "?")
        where = f"p.{src['page']}" if src.get("page") else src.get("section", "")
        locs = groups.setdefault(name, [])
        if where and where not in locs:
            locs.append(where)
    if not groups:
        return
    rows = []
    for name, locs in groups.items():
        # Pages in numeric order; sections keep retrieval order
        locs.sort(key=lambda l: int(l[2:]) if l.startswith("p.") and l[2:].isdigit() else 10**6)
        # File names can come from uploads, so escape everything before rendering as HTML
        pills = "".join(f"<span class='ka-loc'>{html.escape(l)}</span>" for l in locs)
        rows.append(
            f"<div class='ka-cite'><b>{_ext_badge(name)}</b><span class='ka-file'>{html.escape(name)}</span>{pills}</div>"
        )
    st.html(f"<div class='ka-cites' role='list' aria-label='Sources'>{''.join(rows)}</div>")


def _render_trace(tool_trace: list[dict], sources: list[dict]) -> None:
    """Expandable list of the MCP tool calls behind an answer."""
    if not tool_trace:
        return
    n = len(tool_trace)
    rows = []
    for call in tool_trace:
        args = call.get("input")
        args_txt = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) and args else ""
        rows.append(
            f"<div class='row'><span class='tag'>mcp &rarr;</span><span><b>{html.escape(call.get('tool', '?'))}</b> "
            f"<span class='args'>{html.escape(args_txt)}</span></span></div>"
        )
    if sources:
        docs = len({s.get("source") for s in sources})
        rows.append(
            f"<div class='row'><span class='tag'>mcp &larr;</span><span>{len(sources)} passages from "
            f"{docs} document{'s' if docs != 1 else ''}</span></div>"
        )
    with st.expander(f"Retrieval trace · {n} MCP call{'s' if n != 1 else ''}", icon=":material/account_tree:"):
        st.html(f"<div class='ka-trace'>{''.join(rows)}</div>")


def _render_answer_meta(msg: dict) -> None:
    _render_citations(msg.get("sources", []))
    _render_trace(msg.get("tool_trace", []), msg.get("sources", []))


def _render_user_text(text: str) -> None:
    st.html(f"<div class='ka-user-mark'>{html.escape(text)}</div>")


def _meter_html() -> str:
    left = max(0, LIMITS.questions_per_session - st.session_state.usage.questions)
    pct = 100 * left / max(1, LIMITS.questions_per_session)
    return (
        f"<div class='ka-meter'>{left} of {LIMITS.questions_per_session} questions left this session"
        f"<div class='bar'><i style='width:{pct:.0f}%'></i></div></div>"
    )


# ─── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    backend = "OpenSearch Serverless" if VECTOR_BACKEND == "opensearch" else "ChromaDB"
    st.html(
        f"<div class='ka-brand'><div class='ka-mark' aria-hidden='true'></div><div>"
        f"<div class='ka-brand-name'>Knowledge Agent</div>"
        f"<div class='ka-brand-sub'>RAG · Claude · MCP</div></div></div>"
    )

    sources = list_sources()
    rows = "".join(
        f"<div class='ka-doc{' up' if s.startswith('upload_') else ''}'><b>{_ext_badge(s)}</b>{html.escape(s)}</div>"
        for s in sources
    ) or "<p class='ka-hint'>Nothing indexed yet. Upload a document below.</p>"
    st.html(
        f"<div class='ka-side-label'>Indexed documents <span>{len(sources)}</span></div>"
        f"<div class='ka-docs'>{rows}</div>"
    )

    st.html("<div class='ka-side-label'>Add a document</div>")
    if DEMO_MODE:
        st.html(
            f"<p class='ka-hint'>Uploads are visible to other visitors and removed after "
            f"{max(1, LIMITS.upload_ttl_seconds // 60)} min. Public, non-sensitive files only, "
            f"up to {LIMITS.max_upload_bytes // 1000} KB.</p>"
        )
    uploaded = st.file_uploader(
        "Upload PDF, Markdown, CSV, or TXT files",
        type=["pdf", "md", "markdown", "csv", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded and st.button("Index uploaded files", type="primary", icon=":material/upload_file:", use_container_width=True):
        upload_dir = docs_dir() / UPLOADS_SUBDIR
        upload_dir.mkdir(parents=True, exist_ok=True)
        reports = []
        with st.spinner("Indexing through the MCP ingest_document tool..."):
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
        st.info(st.session_state.flash, icon=":material/check_circle:")
        st.session_state.flash = None

    st.html("<div class='ka-side-label'>Session</div>")
    if st.button("New conversation", icon=":material/restart_alt:", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.session is not None:
            st.session_state.session.reset()
        st.rerun()
    st.html(
        f"<div class='ka-links'><a href='{REPO_URL}' target='_blank' rel='noopener'>Source code &#8599;</a>"
        f"<a href='{PROJECT_URL}' target='_blank' rel='noopener'>Case study &#8599;</a></div>"
    )


# ─── Main ────────────────────────────────────────────────────────────
PLACEHOLDER = "Ask about company policies, procedures, or documents"
STATUS_HTML = (
    f"<div class='ka-status'><span class='ka-live'><i class='ka-dot' aria-hidden='true'></i>Online</span>"
    f"<span>{html.escape(AGENT_MODEL)}</span><span>MCP tools</span><span>{backend}</span></div>"
)


def _notice_html() -> str:
    if not DEMO_MODE:
        return ""
    return (
        f"<div class='ka-notice'><span>Public demo on fictional documents for a made-up company.</span>"
        f"{_meter_html()}</div>"
    )


# A question typed on the welcome screen (or an example card) is parked in
# `pending` and the app reruns straight into the conversation layout, so only
# one chat input exists on screen at a time.
pending = st.session_state.pop("pending", None)

if not st.session_state.messages and pending is None:
    # Welcome: everything centered around the input, like a new Claude chat
    with st.container(key="welcome"):
        st.html(
            STATUS_HTML
            + "<h1 class='ka-title'>Ask the <span>knowledge base</span></h1>"
            "<p class='ka-sub'>Answers come only from the indexed documents, with the file and page behind "
            "every claim. Open the retrieval trace under an answer to see the MCP tool calls.</p>"
        )
        typed = st.chat_input(PLACEHOLDER, key="welcome_input")
        with st.container(key="examples"):
            cols = st.columns(len(EXAMPLE_QUESTIONS))
            clicked = None
            for col, example in zip(cols, EXAMPLE_QUESTIONS):
                if col.button(example, use_container_width=True):
                    clicked = example
        if DEMO_MODE:
            st.html(_notice_html())
    if typed or clicked:
        st.session_state.pending = typed or clicked
        st.rerun()
    st.stop()

# Conversation: compact header, messages, input docked at the bottom
header_slot = st.empty()
header_slot.html(f"<div class='ka-convo-head'>{STATUS_HTML}{_notice_html()}</div>")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else AGENT_AVATAR):
        if msg["role"] == "user":
            _render_user_text(msg["content"])
        else:
            st.markdown(msg["content"])
            _render_answer_meta(msg)

prompt = st.chat_input(PLACEHOLDER, key="dock_input") or pending

if prompt:
    refusal = (
        check_question(prompt, st.session_state.usage, _daily_counter(), LIMITS) if DEMO_MODE else None
    )
    if refusal:
        st.warning(refusal, icon=":material/hourglass_top:")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            _render_user_text(prompt)

        with st.chat_message("assistant", avatar=AGENT_AVATAR):
            placeholder = st.empty()
            placeholder.html(
                "<div class='ka-thinking'><span class='ka-dots' aria-hidden='true'><i></i><i></i><i></i></span>"
                "Searching the documents through MCP</div>"
            )
            try:
                result = _get_session().ask(prompt)
            except Exception as e:
                placeholder.error(f"Something went wrong: {e}", icon=":material/error:")
                st.stop()
            placeholder.markdown(result["answer"])
            msg = {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "tool_calls": result["tool_calls"],
                "tool_trace": result.get("tool_trace", []),
            }
            _render_answer_meta(msg)
            st.session_state.messages.append(msg)
        header_slot.html(f"<div class='ka-convo-head'>{STATUS_HTML}{_notice_html()}</div>")
