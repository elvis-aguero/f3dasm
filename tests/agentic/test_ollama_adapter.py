"""Tests for OllamaAdapter — stub out langchain_ollama.ChatOllama."""
import sys
import types


def _install_fake_ollama(response_text: str = "ok"):
    """Install a fake langchain_ollama module that returns response_text."""
    class FakeResponse:
        content = response_text

    class FakeChatOllama:
        def __init__(self, model):
            self.model = model
        def invoke(self, messages):
            return FakeResponse()

    fake_mod = types.ModuleType("langchain_ollama")
    fake_mod.ChatOllama = FakeChatOllama
    sys.modules["langchain_ollama"] = fake_mod
    return FakeChatOllama


def test_ollama_adapter_invoke_returns_string():
    """OllamaAdapter.invoke returns a str."""
    _install_fake_ollama("hello from ollama")

    # Force reimport to pick up the fake module
    if "f3dasm._src.agentic.backends.ollama" in sys.modules:
        del sys.modules["f3dasm._src.agentic.backends.ollama"]

    from f3dasm._src.agentic.backends.ollama import OllamaAdapter

    adapter = OllamaAdapter("llama3", "You are helpful.")
    result = adapter.invoke([{"role": "user", "content": "Hello"}])
    assert result == "hello from ollama"


def test_ollama_adapter_prepends_system_prompt():
    """OllamaAdapter passes SystemMessage as first message."""
    messages_received = []

    class FakeResponse:
        content = "ok"

    class CaptureChatOllama:
        def __init__(self, model):
            pass
        def invoke(self, messages):
            messages_received.extend(messages)
            return FakeResponse()

    fake_mod = sys.modules.get("langchain_ollama", types.ModuleType("langchain_ollama"))
    fake_mod.ChatOllama = CaptureChatOllama
    sys.modules["langchain_ollama"] = fake_mod

    if "f3dasm._src.agentic.backends.ollama" in sys.modules:
        del sys.modules["f3dasm._src.agentic.backends.ollama"]

    from f3dasm._src.agentic.backends.ollama import OllamaAdapter
    from langchain_core.messages import SystemMessage

    adapter = OllamaAdapter("llama3", "Be concise.")
    adapter.invoke([{"role": "user", "content": "Hi"}])

    assert isinstance(messages_received[0], SystemMessage)
    assert messages_received[0].content == "Be concise."


def test_ollama_adapter_skips_system_role_in_messages():
    """System-role dicts in messages are ignored (not doubled)."""
    messages_received = []

    class FakeResponse:
        content = "ok"

    class CaptureChatOllama:
        def __init__(self, model):
            pass
        def invoke(self, messages):
            messages_received.extend(messages)
            return FakeResponse()

    fake_mod = sys.modules.get("langchain_ollama", types.ModuleType("langchain_ollama"))
    fake_mod.ChatOllama = CaptureChatOllama
    sys.modules["langchain_ollama"] = fake_mod

    if "f3dasm._src.agentic.backends.ollama" in sys.modules:
        del sys.modules["f3dasm._src.agentic.backends.ollama"]

    from f3dasm._src.agentic.backends.ollama import OllamaAdapter
    from langchain_core.messages import SystemMessage

    adapter = OllamaAdapter("llama3", "sys")
    adapter.invoke([
        {"role": "system", "content": "ignore me"},
        {"role": "user", "content": "Hi"},
    ])

    # Only one SystemMessage (from system_prompt), not two
    system_msgs = [m for m in messages_received if isinstance(m, SystemMessage)]
    assert len(system_msgs) == 1
