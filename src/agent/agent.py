"""
Agent builder.

Returns a LangChain `AgentExecutor` backed by Claude Sonnet, wired to the
three tools in `tools.py`. Conversation memory is per-Session so multiple
users / threads don't trample each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

try:
    # LangChain 1.x: AgentExecutor + tool-calling agent live in langchain_classic
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
except ImportError:  # LangChain 0.x fallback
    from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import AGENT_MAX_ITERATIONS, AGENT_MODEL, AGENT_TEMPERATURE, MEMORY_WINDOW, check_api_keys
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools import get_document_summary, list_documents, search_knowledge_base


def _build_executor() -> AgentExecutor:
    """Construct the agent + executor. Separate function for test injection."""
    check_api_keys()
    llm = ChatGoogleGenerativeAI(
        model=AGENT_MODEL,
        temperature=AGENT_TEMPERATURE,
        max_output_tokens=2048,
    )
    tools = [search_knowledge_base, list_documents, get_document_summary]
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


def build_agent() -> AgentExecutor:
    """Public factory."""
    return _build_executor()


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
        """
        result = self.executor.invoke({"input": question, "chat_history": self._history})
        answer_text = _coerce_text(result["output"])

        # Update history (keep last N turns)
        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer_text))
        # window measured in turn pairs, so keep 2 * window messages
        if len(self._history) > 2 * self.window:
            self._history = self._history[-2 * self.window :]

        # Extract sources and tool names from intermediate steps
        tool_calls: List[str] = []
        sources: List[Dict[str, Any]] = []
        for action, observation in result.get("intermediate_steps", []):
            tool_calls.append(getattr(action, "tool", "?"))
            if getattr(action, "tool", None) == "search_knowledge_base":
                sources.extend(_parse_source_tags(str(observation)))

        return {
            "answer": answer_text,
            "sources": sources,
            "tool_calls": tool_calls,
        }


def _coerce_text(output: Any) -> str:
    """
    Normalize an AgentExecutor `output` into a plain string.

    Gemini (and other providers) can return content as a list of parts
    (e.g. [{'type': 'text', 'text': '...'}, '...']) instead of a raw
    string. Flatten that into clean text for the UI.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        pieces: List[str] = []
        for part in output:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    pieces.append(text)
        return "".join(pieces).strip()
    return str(output)


def _parse_source_tags(blob: str) -> List[Dict[str, Any]]:
    """
    Parse the bracketed source tags from a retrieval output blob into
    structured dicts — used by the UI to render citations.
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
