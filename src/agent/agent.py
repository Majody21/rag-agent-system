"""
Agent builder.

Returns a LangChain `AgentExecutor` backed by Claude. Its tools come from
the MCP server (search_documents, list_sources, get_document) through the
MCP client in `mcp_client.py`; the agent never imports retrieval code.
Conversation memory is per-Session so multiple users don't trample each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

try:
    # LangChain 1.x: AgentExecutor + tool-calling agent live in langchain_classic
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
except ImportError:  # LangChain 0.x fallback
    from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import AGENT_MAX_ITERATIONS, AGENT_MAX_TOKENS, AGENT_MODEL, MEMORY_WINDOW, check_llm_key
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.mcp_client import get_toolbox


# Models that support server-side refusal fallbacks. On a policy decline the
# API re-runs the same request on a fallback model inside the same call.
_FALLBACK_MODELS = ("claude-opus-5", "claude-fable-5")


def build_llm(model: str = AGENT_MODEL) -> ChatAnthropic:
    """Claude chat model. Sampling params are omitted: current models reject them."""
    check_llm_key()
    kwargs: Dict[str, Any] = {}
    if model.startswith(_FALLBACK_MODELS):
        kwargs = {
            "betas": ["server-side-fallback-2026-07-01"],
            "model_kwargs": {"extra_body": {"fallbacks": "default"}},
        }
    return ChatAnthropic(model=model, max_tokens=AGENT_MAX_TOKENS, **kwargs)


def _build_executor(tools: List[Any] | None = None) -> AgentExecutor:
    """Construct the agent + executor. `tools` defaults to the MCP server's read tools."""
    llm = build_llm()
    if tools is None:
        tools = get_toolbox().agent_tools
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=AGENT_MAX_ITERATIONS,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=False,
    )


def build_agent(tools: List[Any] | None = None) -> AgentExecutor:
    """Public factory."""
    return _build_executor(tools)


@dataclass
class Session:
    """
    A conversational session with sliding-window memory.

    LangChain's memory classes are being deprecated; a simple in-memory
    list of (human, ai) turns with manual truncation is cleaner for our
    scope and does not rely on deprecated APIs.
    """

    executor: AgentExecutor = field(default_factory=build_agent)
    window: int = MEMORY_WINDOW
    _history: List[Any] = field(default_factory=list)

    def reset(self) -> None:
        self._history.clear()

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Run one turn. Returns a dict with:
          - answer: str
          - sources: list[dict]  (retrieved docs across all tool calls)
          - tool_calls: list[str] (names of tools invoked)
          - tool_trace: list[dict] (each MCP tool call with its arguments)
          - usage: dict  (input/output tokens per model, summed over the turn)
        """
        # MCP tools are async and live on the toolbox's event loop, so the
        # whole agent turn runs there.
        usage_cb = UsageMetadataCallbackHandler()
        result = get_toolbox().run(
            self.executor.ainvoke(
                {"input": question, "chat_history": self._history},
                config={"callbacks": [usage_cb]},
            ),
            timeout=600,
        )
        answer_text = _coerce_text(result["output"]) or (
            "I couldn't produce an answer for that request. Try rephrasing it."
        )

        # Update history (keep last N turns)
        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer_text))
        # window measured in turn pairs, so keep 2 * window messages
        if len(self._history) > 2 * self.window:
            self._history = self._history[-2 * self.window :]

        # Extract sources and tool names from intermediate steps
        tool_calls: List[str] = []
        tool_trace: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        for action, observation in result.get("intermediate_steps", []):
            tool_calls.append(getattr(action, "tool", "?"))
            tool_trace.append({"tool": getattr(action, "tool", "?"), "input": getattr(action, "tool_input", None)})
            if getattr(action, "tool", None) == "search_documents":
                sources.extend(_parse_source_tags(_coerce_text(observation)))

        return {
            "answer": answer_text,
            "sources": sources,
            "tool_calls": tool_calls,
            "tool_trace": tool_trace,
            "usage": dict(usage_cb.usage_metadata),
        }


def _coerce_text(output: Any) -> str:
    """
    Normalize model output or an MCP tool result into a plain string.

    Claude and MCP tools both return content as a list of blocks
    (thinking, text, ...) instead of a raw string. Keep only the text.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        pieces: List[str] = []
        for part in output:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    pieces.append(text)
        return "".join(pieces).strip()
    return str(output)


def _parse_source_tags(blob: str) -> List[Dict[str, Any]]:
    """
    Parse the bracketed source tags from a retrieval output blob into
    structured dicts, used by the UI to render citations.
    """
    import re

    results: List[Dict[str, Any]] = []
    for match in re.finditer(r"\[(source=[^\]]+)\]", blob):
        kv_part = match.group(1)
        entry: Dict[str, Any] = {}
        for piece in kv_part.split(", "):
            if "=" in piece:
                k, v = piece.split("=", 1)
                entry[k.strip()] = v.strip().strip("'\"")
        results.append(entry)
    return results
