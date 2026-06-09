"""Claude-SDK-backed adapter for the f3dasm LangGraph agentic runtime."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect as _inspect
import threading
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


_TYPE_MAP: dict = {
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    dict: {"type": "object"},
    list: {"type": "array"},
    str: {"type": "string"},
}


def _infer_schema_from_callable(fn: Any) -> dict:
    """Build a JSON schema dict from a Python callable's type annotations."""
    sig = _inspect.signature(fn)
    props: dict = {}
    required: list[str] = []
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
        if role in ("human", "user"):
            prefix = "Human"
        elif role in ("ai", "assistant"):
            prefix = "Assistant"
        else:
            continue  # skip system messages — passed via system_prompt
        parts.append(f"{prefix}: {content}")
    return "\n\n".join(parts)


class ClaudeAdapter:
    """Wraps claude-agent-sdk; runs one agent turn and returns assistant text.

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
        extra_mcp_servers: dict | None = None,
        extra_allowed_tools: list[str] | None = None,
        persistent: bool = False,
        max_history_pairs: int = 5,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.study_dir = Path(study_dir) if study_dir else None
        self.native_tools = list(native_tools or [])
        self.closure_tools = dict(closure_tools or {})
        self.extra_mcp_servers: dict = dict(extra_mcp_servers or {})
        self.extra_allowed_tools: list[str] = list(extra_allowed_tools or [])
        # persistent and max_history_pairs kept for backward compatibility with
        # Agent subclasses and tests that read these attributes; not used
        # in the core invocation path (history is demand-driven via DelegationLog).
        self.persistent: bool = persistent
        self.max_history_pairs: int = max_history_pairs
        # Lock serializes concurrent delegations to the same shared adapter.
        self._lock: threading.Lock = threading.Lock()
        # Set by StrategizerNode; when truthy, the generator is closed after
        # the next AssistantMessage so the session ends on a routing decision.
        self.route_watcher: Any = None
        # Populated after each ainvoke() with token counts from ResultMessage.
        self.last_usage: dict = {}

    def copy(self) -> "ClaudeAdapter":
        """Always return self.

        Concurrent delegations share this adapter instance and are serialized
        via _lock in invoke(). Episodic memory is demand-driven via RecallHistory
        (backed by DelegationLog) rather than per-adapter history injection.
        """
        return self

    async def ainvoke(self, messages: list[dict]) -> str:
        """Run one agent turn asynchronously; return assembled text."""
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
                            "content": [
                                {"type": "text", "text": f"ERROR: {exc}"}
                            ],
                            "is_error": True,
                        }
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    str(result) if result is not None else ""
                                ),
                            }
                        ]
                    }

                sdk_tools.append(
                    SdkMcpTool(
                        name=tool_name,
                        description=(
                            (fn.__doc__ or tool_name).split("\n")[0].strip()
                        ),
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

        # Merge external stdio MCP servers declared by the Agent subclass.
        if self.extra_mcp_servers:
            mcp_servers.update(self.extra_mcp_servers)

        _base_disallowed = ["WebSearch", "WebFetch", "Task", "ExitPlanMode", "computer"]
        _effective_disallowed = [t for t in _base_disallowed if t not in self.extra_allowed_tools]

        # Non-blocking raw-oracle nudge: a PostToolUse hook that injects a
        # reminder (capped per delegation = per ainvoke) when a Bash/Write
        # call reaches the oracle directly instead of via get_evaluator().
        # Best-effort — if the SDK hook API is unavailable, run without it.
        _hooks = None
        try:
            from claude_agent_sdk import HookMatcher

            from .base import OracleNudgeBudget
            _nudge = OracleNudgeBudget()
            # Expose on the adapter so the runtime can drain + log its
            # firings as direct evidence (see _record_intervention).
            self._oracle_nudge = _nudge

            async def _oracle_hook(input_data, tool_use_id, context):
                msg = _nudge.check(
                    input_data.get("tool_name", ""),
                    input_data.get("tool_input") or {},
                )
                if not msg:
                    return {}
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": msg,
                    }
                }

            _hooks = {"PostToolUse": [HookMatcher(hooks=[_oracle_hook])]}
        except Exception:  # noqa: BLE001 — nudge is best-effort, never fatal
            _hooks = None

        options = ClaudeAgentOptions(
            system_prompt=self.system_prompt,
            model=self.model,
            cwd=str(self.study_dir) if self.study_dir else None,
            tools=self.native_tools or [],
            mcp_servers=mcp_servers if mcp_servers else {},
            allowed_tools=(
                (qualified_mcp_tools + self.native_tools + self.extra_allowed_tools)
                or None
            ),
            disallowed_tools=_effective_disallowed,
            permission_mode="bypassPermissions",
            strict_mcp_config=bool(mcp_servers) or bool(self.extra_mcp_servers),
            **({"hooks": _hooks} if _hooks else {}),
        )

        prompt_str = _format_messages_as_prompt(messages)

        last_assistant = None
        last_result: Any = None
        gen = query(prompt=prompt_str, options=options)
        try:
            async for msg in gen:
                if isinstance(msg, AssistantMessage):
                    last_assistant = msg
                    if self.route_watcher and self.route_watcher():
                        break
                elif isinstance(msg, ResultMessage):
                    last_result = msg
                    break
        finally:
            aclose = getattr(gen, "aclose", None)
            if aclose:
                try:
                    await aclose()
                except Exception:
                    pass

        # Capture token usage from ResultMessage for run-level accounting.
        if last_result is not None:
            self.last_usage = {
                **(last_result.usage or {}),
                "total_cost_usd": last_result.total_cost_usd,
            }
        else:
            self.last_usage = {}

        text = ""
        if last_assistant is not None:
            for block in last_assistant.content:
                if isinstance(block, TextBlock):
                    text += block.text
        return text

    def invoke(self, messages: list[dict]) -> str:
        """Synchronous wrapper around :meth:`ainvoke`.

        Acquires _lock to serialize concurrent callers (e.g. parallel
        delegations to the same shared worker adapter). Transient API/network
        failures are retried with exponential backoff (see retry_on_transient).
        """
        from .base import retry_on_transient
        with self._lock:
            return retry_on_transient(
                lambda: _run_async_safe(self.ainvoke(messages))
            )
