"""Claude-SDK-backed adapter for the f3dasm LangGraph agentic runtime."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect as _inspect
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


def _infer_schema_from_callable(fn: Any) -> dict:
    """Build a JSON schema dict from a Python callable's type annotations."""
    sig = _inspect.signature(fn)
    props: dict = {}
    required: list[str] = []
    _TYPE_MAP = {
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        dict: {"type": "object"},
        list: {"type": "array"},
        str: {"type": "string"},
    }
    for pname, param in sig.parameters.items():
        ann = param.annotation
        if ann is _inspect.Parameter.empty:
            json_type = {"type": "string"}
        else:
            json_type = _TYPE_MAP.get(ann, {"type": "string"})
        props[pname] = json_type
        if param.default is _inspect.Parameter.empty:
            required.append(pname)
    schema: dict = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def _run_async_safe(coro: Any) -> Any:
    """Run *coro* safely whether or not an event loop is already running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def _format_messages_as_prompt(messages: list[dict]) -> str:
    """Convert LangChain-style message dicts to a plain conversation string."""
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        prefix = "Human" if role in ("human", "user") else "Assistant"
        parts.append(f"{prefix}: {content}")
    return "\n\n".join(parts)


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
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            SdkMcpTool,
            TextBlock,
            create_sdk_mcp_server,
            query,
        )

        # Build MCP server from closure_tools if any
        mcp_servers: dict = {}
        qualified_mcp_tools: list[str] = []
        if self.closure_tools:
            server_name = "f3dasm_agent_tools"
            sdk_tools: list[Any] = []
            for tool_name, fn in self.closure_tools.items():
                schema = _infer_schema_from_callable(fn)

                async def _handler(args: dict, bound_fn: Any = fn) -> dict:
                    try:
                        result = bound_fn(**args)
                    except Exception as exc:
                        return {
                            "content": [{"type": "text", "text": f"ERROR: {exc}"}],
                            "is_error": True,
                        }
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": str(result) if result is not None else "",
                            }
                        ]
                    }

                sdk_tools.append(
                    SdkMcpTool(
                        name=tool_name,
                        description=(fn.__doc__ or tool_name).split("\n")[0].strip(),
                        input_schema=schema,
                        handler=_handler,
                    )
                )

            mcp_cfg = create_sdk_mcp_server(
                name=server_name, tools=sdk_tools or None
            )
            mcp_servers = {server_name: mcp_cfg}
            qualified_mcp_tools = [
                f"mcp__{server_name}__{t.name}" for t in sdk_tools
            ]

        options = ClaudeAgentOptions(
            system_prompt=self.system_prompt,
            model=self.model,
            cwd=str(self.study_dir) if self.study_dir else None,
            tools=self.native_tools or [],
            mcp_servers=mcp_servers if mcp_servers else {},
            allowed_tools=(
                (qualified_mcp_tools + self.native_tools)
                if self.native_tools or qualified_mcp_tools
                else []
            ),
            disallowed_tools=[
                "WebSearch",
                "WebFetch",
                "Task",
                "ExitPlanMode",
                "computer",
            ],
            permission_mode="bypassPermissions",
            strict_mcp_config=bool(mcp_servers),
        )

        prompt_str = _format_messages_as_prompt(messages)

        last_assistant = None
        gen = query(prompt=prompt_str, options=options)
        try:
            async for msg in gen:
                if isinstance(msg, AssistantMessage):
                    last_assistant = msg
                elif isinstance(msg, ResultMessage):
                    break
        finally:
            aclose = getattr(gen, "aclose", None)
            if aclose:
                try:
                    await aclose()
                except Exception:
                    pass

        text = ""
        if last_assistant is not None:
            for block in last_assistant.content:
                if isinstance(block, TextBlock):
                    text += block.text
        return text

    def invoke(self, messages: list[dict]) -> str:
        """Synchronous wrapper around :meth:`ainvoke`."""
        return _run_async_safe(self.ainvoke(messages))
