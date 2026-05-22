"""Ollama-backed adapter for the f3dasm LangGraph agentic runtime."""

from __future__ import annotations

from typing import Any

__all__ = ["OllamaAdapter"]


def _to_lc_messages(messages: list[dict]) -> list:
    """Convert LangChain-style message dicts to LangChain message objects."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    result = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in content
            )
        if role in ("human", "user"):
            result.append(HumanMessage(content=content))
        elif role in ("ai", "assistant"):
            result.append(AIMessage(content=content))
        # skip system — passed via system_prompt in OllamaAdapter.__init__
    return result


class OllamaAdapter:
    """Wraps langchain_ollama.ChatOllama for single-turn text generation.

    Ollama does not support MCP or the claude-agent-sdk tool-execution loop.
    The adapter converts message dicts to LangChain messages, prepends the
    system prompt, and returns the raw text response.

    Parameters
    ----------
    model : str
        Ollama model identifier (e.g. ``"llama3"``).
    system_prompt : str
        System prompt prepended to every conversation.
    study_dir : path-like or None
        Unused for Ollama (no cwd support). Accepted for API parity.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str,
        study_dir: Any = None,
    ) -> None:
        from langchain_ollama import ChatOllama

        self.model = model
        self.system_prompt = system_prompt
        self._llm = ChatOllama(model=model)

    def invoke(self, messages: list[dict]) -> str:
        """Run one text-generation turn; return the assistant response."""
        from langchain_core.messages import SystemMessage

        lc_messages = [SystemMessage(content=self.system_prompt)] + _to_lc_messages(messages)
        response = self._llm.invoke(lc_messages)
        return str(response.content)
