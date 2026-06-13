"""Backend surface-area parity — forward-compatible across ALL backends.

Design principle
----------------
Parity is enforced against ClaudeAdapter (the reference) and derived at runtime
from the BACKEND REGISTRY (backends/registry.py), NOT a hardcoded pair of
classes. Every backend the registry advertises is parametrized into these
tests automatically — so registering a new backend (OpenRouter, vLLM, the next
one) immediately subjects it to the same parity checks with no edit here. A new
public attribute/method on one adapter that the others lack will fail the suite
until it is added everywhere (or allow-listed below with a reason).

What "public surface area" means here
--------------------------------------
- Instance attributes set in __init__ whose name does NOT start with '_'.
- Public methods (callable attrs not starting with '_'), INCLUDING inherited
  (the OpenAI-compatible backends inherit their surface from a shared base).

All construction is mocked — no API keys, no network. The OpenAI-compatible
adapters build their LangGraph agent lazily, so merely constructing them is
inert; ClaudeAdapter is built against a stubbed SDK.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest

from f3dasm._src.agentic.backends.registry import (
    available_backends,
    get_adapter_class,
)

# ---------------------------------------------------------------------------
# Reference backend + explicit, documented allowlists for intentional divergence
# ---------------------------------------------------------------------------

REFERENCE = "claude"

# Public members ClaudeAdapter has that the other backends need NOT provide.
CLAUDE_ONLY: set[str] = {
    "ainvoke",  # async entry point; the OpenAI-compatible backends are sync
}

# Per-backend public members allowed to be EXTRA relative to the reference.
# Empty today — every backend shares Claude's public surface. Add an entry
# (with a comment) only for a deliberate, justified divergence.
BACKEND_EXTRA: dict[str, set[str]] = {}

_NON_REFERENCE = [b for b in available_backends() if b != REFERENCE]


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


def _instantiate(backend: str, **kwargs):
    """Construct any registered backend adapter (mocked — no network/keys)."""
    if backend == "claude":
        _install_minimal_sdk()
        import f3dasm._src.agentic.backends.claude as cmod
        cmod._SDK_AVAILABLE = True
    cls = get_adapter_class(backend)
    defaults = dict(model="m", system_prompt="sys")
    defaults.update(kwargs)
    return cls(**defaults)


def _public_instance_attrs(instance) -> set[str]:
    """Public instance attributes (set in __init__, not starting with '_')."""
    return {k for k in vars(instance) if not k.startswith("_")}


def _public_methods(cls) -> set[str]:
    """Public non-dunder callable attributes on cls, INCLUDING inherited.

    Walks the MRO (via dir) rather than only vars(cls), so a backend that
    inherits its public surface from a shared base is judged on its real
    public API — not only the methods defined on the leaf class."""
    return {
        name for name in dir(cls)
        if not name.startswith("_")
        and callable(getattr(cls, name))
    }


# ---------------------------------------------------------------------------
# Core parity — parametrized over every NON-reference backend in the registry.
# Adding a backend to the registry automatically adds it here.
# ---------------------------------------------------------------------------

def test_registry_advertises_the_expected_backends():
    # Guards against a backend silently dropping out of the parity sweep.
    assert set(available_backends()) >= {
        "claude", "ollama", "openrouter", "vllm"}


@pytest.mark.parametrize("backend", _NON_REFERENCE)
def test_public_instance_attrs_match_reference(backend):
    """Every backend's public instance attrs must match ClaudeAdapter's,
    modulo BACKEND_EXTRA. Fails when a backend adds/omits a public attribute."""
    ref = _instantiate(REFERENCE)
    other = _instantiate(backend)
    ref_attrs = _public_instance_attrs(ref)
    other_attrs = _public_instance_attrs(other)

    missing = ref_attrs - other_attrs
    extra = (other_attrs - ref_attrs) - BACKEND_EXTRA.get(backend, set())

    assert not missing, (
        f"{backend} is missing public attrs that {REFERENCE} has: "
        f"{sorted(missing)}"
    )
    assert not extra, (
        f"{backend} has public attrs {REFERENCE} lacks: {sorted(extra)}\n"
        f"Add them to {REFERENCE} or to BACKEND_EXTRA[{backend!r}] with a reason."
    )


@pytest.mark.parametrize("backend", _NON_REFERENCE)
def test_public_methods_match_reference(backend):
    """Every backend's public methods must match ClaudeAdapter's, modulo the
    CLAUDE_ONLY / BACKEND_EXTRA allowlists."""
    ref = _instantiate(REFERENCE)
    other = _instantiate(backend)
    ref_methods = _public_methods(type(ref))
    other_methods = _public_methods(type(other))

    missing = (ref_methods - other_methods) - CLAUDE_ONLY
    extra = (other_methods - ref_methods) - BACKEND_EXTRA.get(backend, set())

    assert not missing, (
        f"{backend} is missing public methods that {REFERENCE} has: "
        f"{sorted(missing)}\nAdd them to {backend} or to CLAUDE_ONLY with a reason."
    )
    assert not extra, (
        f"{backend} has public methods {REFERENCE} lacks: {sorted(extra)}\n"
        f"Add them to {REFERENCE} or to BACKEND_EXTRA[{backend!r}] with a reason."
    )


# ---------------------------------------------------------------------------
# Runtime-contract spot-checks — parametrized over EVERY registered backend, so
# each new backend is type-checked on the attributes the runtime relies on.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", available_backends())
def test_runtime_contract_types(backend):
    a = _instantiate(
        backend,
        native_tools=["Read"],
        extra_mcp_servers={"my_server": {"command": "uvx"}},
        extra_allowed_tools=["mcp__arxiv__search_papers"],
        persistent=True,
        max_history_pairs=7,
    )
    # closure_tools: a mutable dict the nodes append to after __init__
    assert isinstance(a.closure_tools, dict)
    a.closure_tools["probe"] = lambda: None
    assert "probe" in a.closure_tools
    # token accounting + parallel-delegation identity + routing hook
    assert isinstance(a.last_usage, dict)
    assert a.copy() is a
    assert a.route_watcher is None
    assert isinstance(a._lock, type(threading.Lock()))
    # passthrough fields
    assert isinstance(a.native_tools, list) and a.native_tools == ["Read"]
    assert isinstance(a.extra_mcp_servers, dict)
    assert "my_server" in a.extra_mcp_servers
    assert isinstance(a.extra_allowed_tools, list)
    assert a.extra_allowed_tools == ["mcp__arxiv__search_papers"]
    assert a.persistent is True
    assert a.max_history_pairs == 7
    # the runtime's single entry point + native-tool selector
    assert callable(a.invoke)
    assert callable(a.select_native_tools)


@pytest.mark.parametrize("backend", available_backends())
def test_core_fields_stored(backend, tmp_path):
    a = _instantiate(
        backend, model="x-model", system_prompt="be concise", study_dir=tmp_path)
    assert a.model == "x-model"
    assert a.system_prompt == "be concise"
    assert a.study_dir == tmp_path

    a_none = _instantiate(backend, study_dir=None)
    assert a_none.study_dir is None


@pytest.mark.parametrize("backend", available_backends())
def test_select_native_tools_is_a_classmethod_returning_subset(backend):
    cls = get_adapter_class(backend)
    declared = ["Bash", "Read", "Done", "FollowUp", "Write"]
    native = cls.select_native_tools(declared)
    assert isinstance(native, list)
    assert set(native) <= set(declared)  # only ever a subset of declared tools


# ---------------------------------------------------------------------------
# Edit tool sandboxing (shared by all OpenAI-compatible backends)
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
# Endpoint env var resolution (per-backend BASE_URL_ENV)
# ---------------------------------------------------------------------------


def test_ollama_base_url_reads_env_var(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom-host:9999/v1")
    from f3dasm._src.agentic.backends.ollama import OllamaAdapter
    adapter = OllamaAdapter(model="llama3.2", system_prompt="sys")
    assert adapter._base_url == "http://custom-host:9999/v1"


def test_ollama_base_url_falls_back_to_localhost(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    from f3dasm._src.agentic.backends.ollama import OllamaAdapter
    adapter = OllamaAdapter(model="llama3.2", system_prompt="sys")
    assert adapter._base_url == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# arxiv parity
# ---------------------------------------------------------------------------


def test_literature_agent_has_no_arxiv_mcp_server():
    from f3dasm._src.agentic.agents.literature import LiteratureReviewAgent
    assert "arxiv" not in LiteratureReviewAgent.mcp_servers


def test_literature_agent_arxiv_tools_in_build_closure_tools(tmp_path):
    pytest.importorskip("arxiv")  # skip if not installed
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
    real = sys.modules.pop("arxiv", None)
    monkeypatch.setitem(sys.modules, "arxiv", None)
    try:
        from f3dasm._src.agentic.backends.ollama import _build_arxiv_closures
        result = _build_arxiv_closures()
        assert result == {}
    finally:
        if real is not None:
            sys.modules["arxiv"] = real
        else:
            sys.modules.pop("arxiv", None)
