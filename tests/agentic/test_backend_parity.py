"""Backend surface-area parity: OllamaAdapter must expose exactly the same
public interface as ClaudeAdapter at all times — not just as of a fixed date.

Design principle
----------------
These tests do NOT hardcode a list of attributes that were known to exist on
a certain date.  Instead they derive the expected surface area from
ClaudeAdapter itself at import time.  That way, whenever a developer adds a
new public attribute or method to ClaudeAdapter they are immediately told
that OllamaAdapter needs the same addition — and vice-versa.

What "public surface area" means here
--------------------------------------
- All instance attributes set in __init__ whose name does NOT start with '_'.
- All public methods (callable attrs whose name does not start with '_'),
  excluding dunder methods.

Constructor args that differ legitimately (base_url for Ollama, ainvoke for
Claude) are explicitly allowed in an OLLAMA_ONLY / CLAUDE_ONLY allowlist
at the top of this file.  Any new divergence requires a conscious decision
to extend that allowlist with a comment explaining why.
"""
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Explicit allowlists for *known* intentional divergence.
# If you need to add something here, add a comment explaining why.
# ---------------------------------------------------------------------------

# Attributes/methods present on OllamaAdapter but NOT on ClaudeAdapter.
# _base_url is Ollama-specific (endpoint URL); _agent is the lazy LangGraph
# agent; _invoke_once is the internal helper split needed by _build_agent.
OLLAMA_ONLY: set[str] = {
    "_base_url",   # Ollama endpoint; Claude backend has no equivalent concept
    "_agent",      # lazy-built LangGraph react agent; internal to Ollama
    "_invoke_once",  # internal helper called by invoke(); not needed by Claude
    "_build_tools",  # internal helper; Claude uses SDK tool loop, not LangChain
    "_build_agent",  # internal helper; Claude uses SDK, not create_react_agent
}

# Attributes/methods present on ClaudeAdapter but NOT on OllamaAdapter.
# ainvoke is the async entry point; Claude has it because the SDK is async.
# OllamaAdapter.invoke() calls _invoke_once directly (sync LangGraph).
CLAUDE_ONLY: set[str] = {
    "ainvoke",   # async entry point; OllamaAdapter is sync-only via LangGraph
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _install_minimal_sdk() -> None:
    """Ensure a minimal fake claude_agent_sdk is present in sys.modules."""
    mod = sys.modules.get("claude_agent_sdk")
    if mod is None:
        mod = types.ModuleType("claude_agent_sdk")
        sys.modules["claude_agent_sdk"] = mod
    for attr in ("AssistantMessage", "ResultMessage", "TextBlock",
                 "SdkMcpTool", "ClaudeAgentOptions", "create_sdk_mcp_server"):
        if not hasattr(mod, attr):
            setattr(mod, attr, object)
    if not hasattr(mod, "query"):
        async def _noop(prompt, options):
            return
            yield
        mod.query = _noop


def _make_claude(**kwargs):
    _install_minimal_sdk()
    import f3dasm._src.agentic.backends.claude as cmod
    cmod._SDK_AVAILABLE = True
    defaults = dict(model="claude-3", system_prompt="sys")
    defaults.update(kwargs)
    return cmod.ClaudeAdapter(**defaults)


def _make_ollama(**kwargs):
    from f3dasm._src.agentic.backends.ollama import OllamaAdapter
    defaults = dict(model="llama3.2", system_prompt="sys")
    defaults.update(kwargs)
    return OllamaAdapter(**defaults)


def _public_instance_attrs(instance) -> set[str]:
    """Return public instance attributes (set in __init__, not starting with _)."""
    return {k for k in vars(instance) if not k.startswith("_")}


def _public_methods(cls) -> set[str]:
    """Return public non-dunder callable attributes on cls, INCLUDING inherited.

    Walks the MRO (via dir) rather than only vars(cls), so a backend that
    inherits its public surface from a shared base (e.g. the OpenAI-compatible
    adapters) is judged on its real public API — not only the handful of
    methods defined on the leaf class."""
    return {
        name for name in dir(cls)
        if not name.startswith("_")
        and callable(getattr(cls, name))
    }


# ---------------------------------------------------------------------------
# Core parity tests — these grow automatically as the classes grow
# ---------------------------------------------------------------------------


def test_public_instance_attrs_match():
    """Every public instance attribute on ClaudeAdapter must be on OllamaAdapter
    and vice-versa, modulo the explicit OLLAMA_ONLY / CLAUDE_ONLY allowlists.

    This test will FAIL if you add a new public attribute to one adapter
    without adding it to the other (or updating the allowlist with a reason).
    """
    claude = _make_claude()
    ollama = _make_ollama()

    claude_attrs = _public_instance_attrs(claude)
    ollama_attrs = _public_instance_attrs(ollama)

    missing_from_ollama = (claude_attrs - ollama_attrs) - CLAUDE_ONLY
    missing_from_claude = (ollama_attrs - claude_attrs) - OLLAMA_ONLY

    assert not missing_from_ollama, (
        f"OllamaAdapter is missing public attrs that ClaudeAdapter has: "
        f"{sorted(missing_from_ollama)}\n"
        f"Either add them to OllamaAdapter or add to CLAUDE_ONLY with a reason."
    )
    assert not missing_from_claude, (
        f"ClaudeAdapter is missing public attrs that OllamaAdapter has: "
        f"{sorted(missing_from_claude)}\n"
        f"Either add them to ClaudeAdapter or add to OLLAMA_ONLY with a reason."
    )


def test_public_methods_match():
    """Every public method on ClaudeAdapter must be on OllamaAdapter and vice-versa,
    modulo the explicit allowlists.

    This test will FAIL if you add a new public method to one adapter
    without adding it to the other (or updating the allowlist with a reason).
    """
    from f3dasm._src.agentic.backends.ollama import OllamaAdapter
    _install_minimal_sdk()
    import f3dasm._src.agentic.backends.claude as cmod
    cmod._SDK_AVAILABLE = True

    claude_methods = _public_methods(cmod.ClaudeAdapter)
    ollama_methods = _public_methods(OllamaAdapter)

    missing_from_ollama = (claude_methods - ollama_methods) - CLAUDE_ONLY
    missing_from_claude = (ollama_methods - claude_methods) - OLLAMA_ONLY

    assert not missing_from_ollama, (
        f"OllamaAdapter is missing public methods that ClaudeAdapter has: "
        f"{sorted(missing_from_ollama)}\n"
        f"Either add them to OllamaAdapter or add to CLAUDE_ONLY with a reason."
    )
    assert not missing_from_claude, (
        f"ClaudeAdapter is missing public methods that OllamaAdapter has: "
        f"{sorted(missing_from_claude)}\n"
        f"Either add them to ClaudeAdapter or add to OLLAMA_ONLY with a reason."
    )


# ---------------------------------------------------------------------------
# Spot-checks on specific attributes whose *type* matters to the runtime
# (these document expectations callers rely on; they are not exhaustive)
# ---------------------------------------------------------------------------


def test_closure_tools_is_mutable_dict_on_both():
    claude = _make_claude()
    ollama = _make_ollama()
    for adapter, name in [(claude, "ClaudeAdapter"), (ollama, "OllamaAdapter")]:
        assert isinstance(adapter.closure_tools, dict), (
            f"{name}.closure_tools must be a plain dict"
        )
        adapter.closure_tools["probe"] = lambda: None
        assert "probe" in adapter.closure_tools, (
            f"{name}.closure_tools must be mutable after __init__"
        )


def test_last_usage_is_dict_on_both():
    claude = _make_claude()
    ollama = _make_ollama()
    assert isinstance(claude.last_usage, dict), "ClaudeAdapter.last_usage must be a dict"
    assert isinstance(ollama.last_usage, dict), "OllamaAdapter.last_usage must be a dict"


def test_copy_returns_self_on_both():
    """copy() is the no-op identity used by parallel delegation — must return self."""
    claude = _make_claude()
    ollama = _make_ollama()
    assert claude.copy() is claude, "ClaudeAdapter.copy() must return self"
    assert ollama.copy() is ollama, "OllamaAdapter.copy() must return self"


def test_route_watcher_is_none_by_default_on_both():
    """route_watcher is set by StrategizerNode; must start as None."""
    claude = _make_claude()
    ollama = _make_ollama()
    assert claude.route_watcher is None
    assert ollama.route_watcher is None


def test_lock_is_threading_lock_on_both():
    """_lock must be a real threading.Lock so concurrent delegations serialize."""
    claude = _make_claude()
    ollama = _make_ollama()
    assert isinstance(claude._lock, type(threading.Lock())), (
        "ClaudeAdapter._lock must be a threading.Lock"
    )
    assert isinstance(ollama._lock, type(threading.Lock())), (
        "OllamaAdapter._lock must be a threading.Lock"
    )


def test_native_tools_is_list_on_both():
    claude = _make_claude(native_tools=["Read"])
    ollama = _make_ollama(native_tools=["Read"])
    assert isinstance(claude.native_tools, list)
    assert isinstance(ollama.native_tools, list)
    assert claude.native_tools == ["Read"]
    assert ollama.native_tools == ["Read"]


def test_extra_mcp_servers_is_dict_on_both():
    srv = {"my_server": {"command": "uvx", "args": ["my-mcp"]}}
    claude = _make_claude(extra_mcp_servers=srv)
    ollama = _make_ollama(extra_mcp_servers=srv)
    assert isinstance(claude.extra_mcp_servers, dict)
    assert isinstance(ollama.extra_mcp_servers, dict)
    assert "my_server" in claude.extra_mcp_servers
    assert "my_server" in ollama.extra_mcp_servers


def test_extra_allowed_tools_is_list_on_both():
    tools = ["mcp__arxiv__search_papers"]
    claude = _make_claude(extra_allowed_tools=tools)
    ollama = _make_ollama(extra_allowed_tools=tools)
    assert isinstance(claude.extra_allowed_tools, list)
    assert isinstance(ollama.extra_allowed_tools, list)
    assert claude.extra_allowed_tools == tools
    assert ollama.extra_allowed_tools == tools


def test_persistent_flag_stored_on_both():
    claude = _make_claude(persistent=True)
    ollama = _make_ollama(persistent=True)
    assert claude.persistent is True
    assert ollama.persistent is True


def test_max_history_pairs_stored_on_both():
    claude = _make_claude(max_history_pairs=7)
    ollama = _make_ollama(max_history_pairs=7)
    assert claude.max_history_pairs == 7
    assert ollama.max_history_pairs == 7


def test_study_dir_stored_as_path_or_none_on_both(tmp_path):
    claude = _make_claude(study_dir=tmp_path)
    ollama = _make_ollama(study_dir=tmp_path)
    assert claude.study_dir == tmp_path
    assert ollama.study_dir == tmp_path

    claude_none = _make_claude(study_dir=None)
    ollama_none = _make_ollama(study_dir=None)
    assert claude_none.study_dir is None
    assert ollama_none.study_dir is None


def test_model_stored_on_both():
    claude = _make_claude(model="claude-opus-4")
    ollama = _make_ollama(model="llama3.3")
    assert claude.model == "claude-opus-4"
    assert ollama.model == "llama3.3"


def test_system_prompt_stored_on_both():
    claude = _make_claude(system_prompt="be concise")
    ollama = _make_ollama(system_prompt="be concise")
    assert claude.system_prompt == "be concise"
    assert ollama.system_prompt == "be concise"


def test_invoke_is_callable_on_both():
    """invoke() is the runtime's single entry point — must exist and be callable."""
    claude = _make_claude()
    ollama = _make_ollama()
    assert callable(claude.invoke)
    assert callable(ollama.invoke)


# ---------------------------------------------------------------------------
# Edit tool sandboxing (Ollama)
# ---------------------------------------------------------------------------


def test_ollama_edit_tool_rejects_path_outside_workspace(tmp_path):
    from f3dasm._src.agentic.backends.ollama import _make_edit_tool
    tool_fn = _make_edit_tool(tmp_path).func
    result = tool_fn(path="../escape.txt", old_str="x", new_str="y")
    assert result.startswith("ERROR: edit rejected")


def test_ollama_edit_tool_absolute_path_outside_rejected(tmp_path):
    from f3dasm._src.agentic.backends.ollama import _make_edit_tool
    tool_fn = _make_edit_tool(tmp_path).func
    result = tool_fn(path="/etc/hosts", old_str="x", new_str="y")
    assert result.startswith("ERROR: edit rejected")


def test_ollama_edit_tool_traversal_blocked(tmp_path):
    from f3dasm._src.agentic.backends.ollama import _make_edit_tool
    sub = tmp_path / "sub"
    sub.mkdir()
    tool_fn = _make_edit_tool(tmp_path).func
    result = tool_fn(path="sub/../../escape.txt", old_str="x", new_str="y")
    assert result.startswith("ERROR: edit rejected")


def test_ollama_edit_tool_accepts_path_inside_workspace(tmp_path):
    from f3dasm._src.agentic.backends.ollama import _make_edit_tool
    target = tmp_path / "sub" / "file.txt"
    target.parent.mkdir()
    target.write_text("hello world")
    tool_fn = _make_edit_tool(tmp_path).func
    result = tool_fn(path="sub/file.txt", old_str="hello", new_str="goodbye")
    assert "Edited" in result
    assert target.read_text() == "goodbye world"


# ---------------------------------------------------------------------------
# OLLAMA_BASE_URL env var
# ---------------------------------------------------------------------------


def test_ollama_base_url_reads_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom-host:9999/v1")
    # Rebuild adapter via _make_adapter path — simulate what agent_runtime does
    import os
    from f3dasm._src.agentic.backends.ollama import OllamaAdapter
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    adapter = OllamaAdapter(model="llama3.2", system_prompt="sys", base_url=base_url)
    assert adapter._base_url == "http://custom-host:9999/v1"


def test_ollama_base_url_falls_back_to_localhost(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    import os
    from f3dasm._src.agentic.backends.ollama import OllamaAdapter
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    adapter = OllamaAdapter(model="llama3.2", system_prompt="sys", base_url=base_url)
    assert adapter._base_url == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# arxiv parity
# ---------------------------------------------------------------------------


def test_literature_agent_has_no_arxiv_mcp_server():
    from f3dasm._src.agentic.agents.literature import LiteratureReviewAgent
    assert "arxiv" not in LiteratureReviewAgent.mcp_servers


def test_literature_agent_arxiv_tools_in_build_closure_tools(tmp_path):
    arxiv = pytest.importorskip("arxiv")  # skip if not installed
    from f3dasm._src.agentic.agents.literature import LiteratureReviewAgent
    tools = LiteratureReviewAgent().build_closure_tools(tmp_path)
    expected = {
        "mcp__arxiv__search_papers",
        "mcp__arxiv__list_papers",
        "mcp__arxiv__download_paper",
        "mcp__arxiv__read_paper",
    }
    assert expected.issubset(set(tools.keys()))


def test_build_arxiv_closures_returns_empty_without_package(monkeypatch):
    import sys
    # Temporarily hide the arxiv package
    real = sys.modules.pop("arxiv", None)
    # Also block import
    monkeypatch.setitem(sys.modules, "arxiv", None)
    try:
        # Re-import to pick up the patched sys.modules
        import importlib
        import f3dasm._src.agentic.backends.ollama as om
        importlib.reload(om)
        result = om._build_arxiv_closures()
        assert result == {}
    finally:
        if real is not None:
            sys.modules["arxiv"] = real
        else:
            sys.modules.pop("arxiv", None)
        import importlib
        import f3dasm._src.agentic.backends.ollama as om
        importlib.reload(om)
