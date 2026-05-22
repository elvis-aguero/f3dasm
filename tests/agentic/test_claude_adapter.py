"""Tests for ClaudeAdapter — stub out claude_agent_sdk.query."""
import types
import sys


# ---------------------------------------------------------------------------
# Helpers to build realistic fake SDK event streams
# ---------------------------------------------------------------------------

class _TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _AssistantMessage:
    def __init__(self, blocks) -> None:
        self.content = blocks


class _ResultMessage:
    pass


def make_async_gen_with_messages(*text_contents):
    """Yield AssistantMessage-like events with TextBlock content, then ResultMessage."""
    async def _gen(prompt, options):
        yield _AssistantMessage([_TextBlock(t) for t in text_contents])
        yield _ResultMessage()
    return _gen


def _install_fake_sdk(**extra):
    """Install (or update) a fake claude_agent_sdk in sys.modules."""
    mod = sys.modules.get("claude_agent_sdk")
    if mod is None:
        mod = types.ModuleType("claude_agent_sdk")
        sys.modules["claude_agent_sdk"] = mod

    mod.AssistantMessage = _AssistantMessage
    mod.ResultMessage = _ResultMessage
    mod.TextBlock = _TextBlock
    mod.SdkMcpTool = object  # not used in these tests
    mod.ClaudeAgentOptions = lambda **kw: kw
    mod.create_sdk_mcp_server = lambda name=None, tools=None: {"name": name}

    for k, v in extra.items():
        setattr(mod, k, v)

    return mod


def _get_adapter():
    """Return the (possibly reloaded) ClaudeAdapter with SDK pre-marked available."""
    import f3dasm._src.agentic.backends.claude as cmod
    cmod._SDK_AVAILABLE = True
    return cmod.ClaudeAdapter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_text_block():
    """Single TextBlock → returns that text exactly."""
    _install_fake_sdk(query=make_async_gen_with_messages("hello world"))
    ClaudeAdapter = _get_adapter()
    adapter = ClaudeAdapter("claude-3", "sys", None, [])
    result = adapter.invoke([{"role": "user", "content": "hi"}])
    assert result == "hello world"


def test_multiple_text_blocks_concatenated():
    """Multiple TextBlocks in one AssistantMessage → concatenated."""
    _install_fake_sdk(query=make_async_gen_with_messages("foo", "bar", "baz"))
    ClaudeAdapter = _get_adapter()
    adapter = ClaudeAdapter("claude-3", "sys", None, [])
    result = adapter.invoke([{"role": "user", "content": "hi"}])
    assert result == "foobarbaz"


def test_non_text_blocks_are_ignored():
    """Non-TextBlock content blocks are skipped; only TextBlock.text is collected."""

    class _ToolUseBlock:
        """Simulates a tool_use content block — should be ignored."""
        name = "Bash"
        input = {"command": "ls"}

    async def _gen_mixed(prompt, options):
        # AssistantMessage with mixed block types
        yield _AssistantMessage([
            _TextBlock("kept"),
            _ToolUseBlock(),
            _TextBlock(" also kept"),
        ])
        yield _ResultMessage()

    _install_fake_sdk(query=_gen_mixed)
    ClaudeAdapter = _get_adapter()
    adapter = ClaudeAdapter("claude-3", "sys", None, [])
    result = adapter.invoke([{"role": "user", "content": "hi"}])
    assert result == "kept also kept"
