"""Claude-SDK-backed adapter for the f3dasm LangGraph agentic runtime."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

__all__ = ["ClaudeAdapter"]

_SDK_AVAILABLE: bool | None = None  # None = not yet checked


def _require_sdk() -> None:
    global _SDK_AVAILABLE
    if _SDK_AVAILABLE is None:
        try:
            import claude_agent_sdk  # noqa: F401
            _SDK_AVAILABLE = True
        except ImportError:
            _SDK_AVAILABLE = False
    if not _SDK_AVAILABLE:
        raise ImportError(
            "claude-agent-sdk is required for the Claude backend. "
            "Install it with: uv add claude-agent-sdk"
        )


class ClaudeAdapter:
    """Wraps claude-agent-sdk; runs one full agent turn and returns assistant text.

    The SDK handles its own tool-execution loop (Bash, Read, Write, Edit).
    This adapter converts a list of LangChain-style message dicts to the SDK
    format, runs the query, and assembles the final text response.

    Parameters
    ----------
    model : str
        Claude model identifier.
    system_prompt : str
        System prompt for the agent.
    study_dir : Path or None
        Working directory passed to the SDK as ``cwd``.
    native_tools : list[str]
        Tool names to enable (e.g. ``["Bash", "Read", "Write"]``).
    closure_tools : dict[str, callable] or None
        Extra Python callables exposed to the model as MCP tools.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str,
        study_dir: Path | None = None,
        native_tools: list[str] | None = None,
        closure_tools: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.study_dir = Path(study_dir) if study_dir else None
        self.native_tools = list(native_tools or [])
        self.closure_tools = dict(closure_tools or {})

    async def ainvoke(self, messages: list[dict]) -> str:
        """Run one agent turn asynchronously; return assembled assistant text."""
        _require_sdk()
        from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, query

        mcp_servers = []
        if self.closure_tools:
            mcp_servers.append(create_sdk_mcp_server(self.closure_tools))

        options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=self.system_prompt,
            allowed_tools=self.native_tools,
            mcp_servers=mcp_servers,
            cwd=str(self.study_dir) if self.study_dir else None,
        )

        parts: list[str] = []
        async for event in query(messages=messages, options=options):
            content = getattr(event, "content", None)
            if isinstance(content, str):
                parts.append(content)

        return "".join(parts)

    def invoke(self, messages: list[dict]) -> str:
        """Synchronous wrapper around :meth:`ainvoke`."""
        return asyncio.run(self.ainvoke(messages))
