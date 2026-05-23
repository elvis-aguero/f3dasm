"""Tests for OllamaAdapter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_adapter(**kwargs):
    from f3dasm._src.agentic.backends.ollama import OllamaAdapter
    defaults = dict(model="llama3.2", system_prompt="You are helpful.")
    defaults.update(kwargs)
    return OllamaAdapter(**defaults)


# ---------------------------------------------------------------------------
# Interface parity with ClaudeAdapter
# ---------------------------------------------------------------------------


def test_has_closure_tools_dict():
    adapter = _make_adapter()
    assert isinstance(adapter.closure_tools, dict)


def test_closure_tools_mutable_after_init():
    adapter = _make_adapter()
    adapter.closure_tools["foo"] = lambda x: x
    assert "foo" in adapter.closure_tools


# ---------------------------------------------------------------------------
# _build_tools maps native names to LangChain tools
# ---------------------------------------------------------------------------


def test_native_tools_mapped():
    adapter = _make_adapter(native_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"])
    tools = adapter._build_tools()
    names = {t.name for t in tools}
    assert names == {"Bash", "Read", "Write", "Edit", "Glob", "Grep"}


def test_unknown_native_tool_ignored():
    adapter = _make_adapter(native_tools=["Bash", "NonExistent"])
    tools = adapter._build_tools()
    names = {t.name for t in tools}
    assert "NonExistent" not in names
    assert "Bash" in names


def test_closure_tools_become_structured_tools():
    def my_tool(x: str) -> str:
        """A test tool."""
        return x
    adapter = _make_adapter()
    adapter.closure_tools["MyTool"] = my_tool
    tools = adapter._build_tools()
    names = {t.name for t in tools}
    assert "MyTool" in names


# ---------------------------------------------------------------------------
# Glob tool works correctly (pattern-based, not directory listing)
# ---------------------------------------------------------------------------


def test_glob_tool_matches_pattern(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    adapter = _make_adapter(native_tools=["Glob"], study_dir=tmp_path)
    tools = {t.name: t for t in adapter._build_tools()}
    result = tools["Glob"].invoke({"pattern": "*.py"})
    assert "a.py" in result
    assert "b.txt" not in result


# ---------------------------------------------------------------------------
# invoke() uses fresh thread_id per call (no state leakage)
# ---------------------------------------------------------------------------


def test_invoke_uses_fresh_thread_id_each_call():
    thread_ids = []
    fake_result = {"messages": [MagicMock(content="done")]}

    def fake_invoke(state, config=None):
        thread_ids.append(config["configurable"]["thread_id"])
        return fake_result

    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = fake_invoke

    adapter = _make_adapter()
    adapter._agent = fake_agent

    adapter.invoke([{"role": "user", "content": "hello"}])
    adapter.invoke([{"role": "user", "content": "world"}])

    assert len(thread_ids) == 2
    assert thread_ids[0] != thread_ids[1]


# ---------------------------------------------------------------------------
# invoke() returns last message content
# ---------------------------------------------------------------------------


def test_invoke_returns_last_message_content():
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "messages": [
            MagicMock(content="intermediate"),
            MagicMock(content="final answer"),
        ]
    }
    adapter = _make_adapter()
    adapter._agent = fake_agent
    result = adapter.invoke([{"role": "user", "content": "go"}])
    assert result == "final answer"


# ---------------------------------------------------------------------------
# Agent built lazily — closure_tools populated before first invoke
# ---------------------------------------------------------------------------


def test_agent_built_lazily():
    adapter = _make_adapter()
    assert adapter._agent is None
    adapter.closure_tools["Done"] = lambda summary: "done"

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"messages": [MagicMock(content="ok")]}

    with patch.object(adapter, "_build_agent", return_value=fake_agent) as mock_build:
        adapter.invoke([{"role": "user", "content": "hi"}])
        mock_build.assert_called_once()
        adapter.invoke([{"role": "user", "content": "hi again"}])
        mock_build.assert_called_once()  # not rebuilt on second call


def test_done_closure_in_tools_when_agent_built():
    """Closure added before first invoke appears in built tools."""
    route = {}

    def Done(summary: str) -> str:
        """Signal done."""
        route["done"] = summary
        return "done"

    adapter = _make_adapter()
    adapter.closure_tools["Done"] = Done
    tools = adapter._build_tools()
    names = {t.name for t in tools}
    assert "Done" in names
