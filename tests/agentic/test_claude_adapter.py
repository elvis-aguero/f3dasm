"""Tests for ClaudeAdapter — stub out claude_agent_sdk.query."""
import asyncio
import pytest
from unittest.mock import patch


def make_async_gen(*contents):
    """Return an async generator that yields events with .content."""
    async def _gen(messages, options):
        for c in contents:
            class Ev:
                content = c
            yield Ev()
    return _gen


def test_claude_adapter_invoke_returns_string():
    """ClaudeAdapter.invoke returns the assembled assistant text."""
    from f3dasm._src.agentic.backends.claude import ClaudeAdapter

    with patch("f3dasm._src.agentic.backends.claude._require_sdk"):
        with patch("claude_agent_sdk.query", side_effect=make_async_gen("hello")):
            # need to mock the import inside ainvoke
            pass

    # Use monkeypatch approach — stub at module level
    import types, sys
    fake_sdk = types.ModuleType("claude_agent_sdk")
    fake_sdk.query = make_async_gen("hello")
    fake_sdk.ClaudeAgentOptions = lambda **kw: None
    fake_sdk.create_sdk_mcp_server = lambda d: None
    sys.modules["claude_agent_sdk"] = fake_sdk

    from importlib import reload
    import f3dasm._src.agentic.backends.claude as mod
    mod._SDK_AVAILABLE = True

    adapter = mod.ClaudeAdapter("claude-3", "sys", None, [])
    assert adapter.invoke([]) == "hello"


def test_claude_adapter_assembles_multiple_events():
    """ClaudeAdapter concatenates content from multiple events."""
    import types, sys
    from f3dasm._src.agentic.backends.claude import ClaudeAdapter

    import f3dasm._src.agentic.backends.claude as mod
    mod._SDK_AVAILABLE = True
    mod_sdk = sys.modules.get("claude_agent_sdk")
    if mod_sdk is None:
        mod_sdk = types.ModuleType("claude_agent_sdk")
        sys.modules["claude_agent_sdk"] = mod_sdk

    mod_sdk.query = make_async_gen("foo", "bar")
    mod_sdk.ClaudeAgentOptions = lambda **kw: None
    mod_sdk.create_sdk_mcp_server = lambda d: None

    adapter = ClaudeAdapter("claude-3", "sys", None, [])
    assert adapter.invoke([]) == "foobar"


def test_claude_adapter_skips_non_string_content():
    """Events with non-string or missing content are ignored."""
    import types, sys
    from f3dasm._src.agentic.backends.claude import ClaudeAdapter

    async def gen_mixed(messages, options):
        class EvStr:
            content = "ok"
        class EvNone:
            content = None
        class EvMissing:
            pass
        class EvInt:
            content = 42
        yield EvStr()
        yield EvNone()
        yield EvMissing()
        yield EvInt()

    mod_sdk = sys.modules.get("claude_agent_sdk")
    if mod_sdk is None:
        mod_sdk = types.ModuleType("claude_agent_sdk")
        sys.modules["claude_agent_sdk"] = mod_sdk

    mod_sdk.query = gen_mixed
    mod_sdk.ClaudeAgentOptions = lambda **kw: None
    mod_sdk.create_sdk_mcp_server = lambda d: None

    import f3dasm._src.agentic.backends.claude as cmod
    cmod._SDK_AVAILABLE = True
    adapter = ClaudeAdapter("claude-3", "sys", None, [])
    assert adapter.invoke([]) == "ok"
