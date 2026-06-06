"""Extra tests for agent_runtime.py — config loading, budget parsing, and
_make_adapter edge cases."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _load_study_config
# ---------------------------------------------------------------------------


def test_load_study_config_returns_empty_dict_when_no_config(tmp_path):
    """_load_study_config returns {} when config.yaml does not exist."""
    from f3dasm._src.agentic.agent_runtime import _load_study_config

    result = _load_study_config(tmp_path)
    assert result == {}


def test_load_study_config_reads_yaml_when_present(tmp_path):
    """_load_study_config reads and parses config.yaml."""
    import yaml as _yaml
    from f3dasm._src.agentic.agent_runtime import _load_study_config

    config = {"model": "claude-haiku", "budget": 3600}
    (tmp_path / "config.yaml").write_text(_yaml.dump(config))

    result = _load_study_config(tmp_path)
    assert result["model"] == "claude-haiku"
    assert result["budget"] == 3600


def test_load_study_config_returns_empty_for_empty_yaml(tmp_path):
    """_load_study_config returns {} when config.yaml is empty."""
    from f3dasm._src.agentic.agent_runtime import _load_study_config

    (tmp_path / "config.yaml").write_text("")

    result = _load_study_config(tmp_path)
    assert result == {}


# ---------------------------------------------------------------------------
# _parse_budget_str
# ---------------------------------------------------------------------------


def test_parse_budget_str_none():
    """_parse_budget_str returns None for None input."""
    from f3dasm._src.agentic.agent_runtime import _parse_budget_str

    assert _parse_budget_str(None) is None


def test_parse_budget_str_float():
    """_parse_budget_str returns float for numeric input."""
    from f3dasm._src.agentic.agent_runtime import _parse_budget_str

    assert _parse_budget_str(3600.0) == 3600.0
    assert _parse_budget_str(1800) == 1800.0


def test_parse_budget_str_hhmmss():
    """_parse_budget_str parses HH:MM:SS string."""
    from f3dasm._src.agentic.agent_runtime import _parse_budget_str

    result = _parse_budget_str("01:30:00")
    assert result == 5400.0  # 1h30m = 5400s


def test_parse_budget_str_float_string():
    """_parse_budget_str parses a plain float string."""
    from f3dasm._src.agentic.agent_runtime import _parse_budget_str

    result = _parse_budget_str("7200")
    assert result == 7200.0


# ---------------------------------------------------------------------------
# AgenticRun.__init__ reads config.yaml
# ---------------------------------------------------------------------------


def test_agentic_run_reads_model_from_config(tmp_path):
    """AgenticRun reads model from config.yaml if not passed explicitly."""
    import yaml as _yaml
    from f3dasm._src.agentic.agent_runtime import AgenticRun

    (tmp_path / "PROBLEM_STATEMENT.md").write_text("test")
    (tmp_path / "config.yaml").write_text(_yaml.dump({"model": "claude-opus"}))

    run = AgenticRun(tmp_path)
    assert run._model == "claude-opus"


def test_agentic_run_reads_budget_from_config(tmp_path):
    """AgenticRun reads budget from config.yaml if not passed explicitly."""
    import yaml as _yaml
    from f3dasm._src.agentic.agent_runtime import AgenticRun

    (tmp_path / "PROBLEM_STATEMENT.md").write_text("test")
    (tmp_path / "config.yaml").write_text(_yaml.dump({"budget": "02:00:00"}))

    run = AgenticRun(tmp_path)
    assert run._budget == 7200.0


def test_agentic_run_explicit_budget_overrides_config(tmp_path):
    """AgenticRun explicit budget parameter overrides config.yaml value."""
    import yaml as _yaml
    from f3dasm._src.agentic.agent_runtime import AgenticRun

    (tmp_path / "PROBLEM_STATEMENT.md").write_text("test")
    (tmp_path / "config.yaml").write_text(_yaml.dump({"budget": "02:00:00"}))

    run = AgenticRun(tmp_path, budget=1000)
    assert run._budget == 1000


# ---------------------------------------------------------------------------
# AgenticRun.execute raises AgenticRunError when PROBLEM_STATEMENT.md missing
# ---------------------------------------------------------------------------


def test_execute_raises_when_no_problem_statement(tmp_path):
    """AgenticRun.execute raises AgenticRunError when PROBLEM_STATEMENT.md is missing."""
    from f3dasm._src.agentic.agent_runtime import AgenticRun, AgenticRunError

    run = AgenticRun(tmp_path)
    with pytest.raises(AgenticRunError, match="PROBLEM_STATEMENT"):
        run.execute()


# ---------------------------------------------------------------------------
# _make_adapter for ollama backend calls build_closure_tools
# ---------------------------------------------------------------------------


def test_make_adapter_ollama_calls_build_closure_tools(tmp_path):
    """_make_adapter for ollama backend calls agent.build_closure_tools."""
    from f3dasm._src.agentic.agent_runtime import AgenticRun, _default_graph, DEFAULT_MODEL
    from f3dasm._src.agentic.backends.base import Agent

    (tmp_path / "PROBLEM_STATEMENT.md").write_text("test")

    run = AgenticRun.__new__(AgenticRun)
    run.study_dir = tmp_path
    run._model = DEFAULT_MODEL
    run._backend = "ollama"
    run._graph_spec = MagicMock()
    run._graph_spec.outgoing.return_value = []

    run_dir = tmp_path / "runs" / "test"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True, exist_ok=True)
    run._run_dir = run_dir

    class TestAgent(Agent):
        description = "Test."
        tools = frozenset({"Bash"})

    agent = TestAgent()

    mock_closures = {"Done": lambda s: "done"}
    with patch.object(agent, "build_closure_tools", return_value=mock_closures) as mock_build:
        with patch("f3dasm._src.agentic.backends.ollama.OllamaAdapter") as MockOllama:
            mock_instance = MagicMock()
            mock_instance.closure_tools = {}
            MockOllama.return_value = mock_instance
            import os
            with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434/v1"}):
                result = run._make_adapter("implementer", agent)

    mock_build.assert_called_once()
    assert "Done" in mock_instance.closure_tools


# ---------------------------------------------------------------------------
# AgenticRun._make_adapter with extra_closures updates adapter
# ---------------------------------------------------------------------------


def test_make_adapter_claude_extra_closures_injected(tmp_path):
    """_make_adapter injects extra_closures from agent.build_closure_tools into adapter."""
    from f3dasm._src.agentic.agent_runtime import AgenticRun, _default_graph, DEFAULT_MODEL
    from f3dasm._src.agentic.backends.base import Agent

    (tmp_path / "PROBLEM_STATEMENT.md").write_text("test")

    run = AgenticRun.__new__(AgenticRun)
    run.study_dir = tmp_path
    run._model = DEFAULT_MODEL
    run._backend = "claude"
    run._graph_spec = MagicMock()
    run._graph_spec.outgoing.return_value = []

    run_dir = tmp_path / "runs" / "test"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True, exist_ok=True)
    run._run_dir = run_dir

    class TestAgent(Agent):
        description = "Test."
        tools = frozenset({"Bash"})

    agent = TestAgent()

    extra_closures = {"SpecialTool": lambda x: x}
    with patch.object(agent, "build_closure_tools", return_value=extra_closures):
        with patch("f3dasm._src.agentic.agent_runtime.ClaudeAdapter") as MockClaude:
            mock_instance = MagicMock()
            mock_instance.closure_tools = {}
            MockClaude.return_value = mock_instance

            result = run._make_adapter("implementer", agent)

    assert "SpecialTool" in mock_instance.closure_tools
