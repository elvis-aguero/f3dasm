"""Tests for OptimizationAgent and related strategizer updates."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Part A — OptimizationAgent prompt content
# ---------------------------------------------------------------------------

def test_optimization_prompt_exploit_framing():
    from f3dasm._src.agentic.agents.optimization import (
        OPTIMIZATION_SYSTEM_PROMPT,
    )
    for token in ("exploit", "surrogate", "get_evaluator"):
        assert token in OPTIMIZATION_SYSTEM_PROMPT, (
            f"OPTIMIZATION_SYSTEM_PROMPT missing token: {token!r}"
        )


def test_optimization_prompt_full_loop_framing():
    from f3dasm._src.agentic.agents.optimization import (
        OPTIMIZATION_SYSTEM_PROMPT,
    )
    # Must state it runs the WHOLE loop internally
    assert "one delegation" in OPTIMIZATION_SYSTEM_PROMPT


def test_optimization_prompt_no_builtin_gp_claim():
    from f3dasm._src.agentic.agents.optimization import (
        OPTIMIZATION_SYSTEM_PROMPT,
    )
    # Must explicitly state f3dasm has no built-in GP (case-insensitive)
    prompt_lower = OPTIMIZATION_SYSTEM_PROMPT.lower()
    assert "no built-in gp" in prompt_lower or \
        "has no built-in gp" in prompt_lower


def test_optimization_prompt_report_header():
    from f3dasm._src.agentic.agents.optimization import (
        OPTIMIZATION_SYSTEM_PROMPT,
    )
    assert "## Report" in OPTIMIZATION_SYSTEM_PROMPT


def test_optimization_prompt_report_sections():
    from f3dasm._src.agentic.agents.optimization import (
        OPTIMIZATION_SYSTEM_PROMPT,
    )
    for section in (
        "### Actions taken",
        "### Files touched",
        "### Conclusions",
        "### Numbers",
    ):
        assert section in OPTIMIZATION_SYSTEM_PROMPT, (
            f"Missing section {section!r} in OPTIMIZATION_SYSTEM_PROMPT"
        )


# ---------------------------------------------------------------------------
# Part A — OptimizationAgent class attributes
# ---------------------------------------------------------------------------

def test_optimization_agent_import():
    from f3dasm.agentic import OptimizationAgent  # noqa: F401


def test_optimization_agent_role():
    from f3dasm.agentic import OptimizationAgent
    assert OptimizationAgent.role == "implementer"


def test_optimization_agent_description_nonempty():
    from f3dasm.agentic import OptimizationAgent
    assert OptimizationAgent.description
    assert len(OptimizationAgent.description) > 10


def test_optimization_agent_description_content():
    from f3dasm.agentic import OptimizationAgent
    desc = OptimizationAgent.description
    assert "surrogate" in desc
    assert "50+" in desc or "~50" in desc


def test_optimization_agent_report_convention():
    from f3dasm.agentic import OptimizationAgent
    assert "## Report" in OptimizationAgent.system_prompt


def test_optimization_agent_report_sections_attr():
    from f3dasm.agentic import OptimizationAgent
    assert "### Actions taken" in OptimizationAgent.report_sections
    assert "### Numbers" in OptimizationAgent.report_sections


def test_optimization_agent_tools():
    from f3dasm.agentic import OptimizationAgent
    assert "Bash" in OptimizationAgent.tools
    assert "ReportEvals" in OptimizationAgent.tools


def test_optimization_agent_reset_on_checkpoint():
    from f3dasm.agentic import OptimizationAgent
    assert OptimizationAgent.reset_on_checkpoint is True


def test_optimization_agent_not_in_default_graph():
    """OptimizationAgent must NOT be wired into _default_graph."""
    from f3dasm._src.agentic.agents._graphs import _default_graph
    from f3dasm.agentic import OptimizationAgent
    graph = _default_graph()
    agent_types = {type(agent) for agent in graph.nodes.values()}
    assert OptimizationAgent not in agent_types


# ---------------------------------------------------------------------------
# Part B — strategizer prompt additions
# ---------------------------------------------------------------------------

def test_strategizer_mentions_specialist_when_connected():
    from f3dasm._src.agentic.agents.strategizer import (
        STRATEGIZER_SYSTEM_PROMPT,
    )
    # Must explain specialists are conditional on being connected
    prompt = STRATEGIZER_SYSTEM_PROMPT
    assert "when connected" in prompt or "WHEN PRESENT" in prompt


def test_strategizer_mentions_data_generator_block2():
    from f3dasm._src.agentic.agents.strategizer import (
        STRATEGIZER_SYSTEM_PROMPT,
    )
    assert "DataGeneratorAgent" in STRATEGIZER_SYSTEM_PROMPT


def test_strategizer_mentions_optimization_agent_block34():
    from f3dasm._src.agentic.agents.strategizer import (
        STRATEGIZER_SYSTEM_PROMPT,
    )
    assert "OptimizationAgent" in STRATEGIZER_SYSTEM_PROMPT


def test_strategizer_does_not_promise_absent_agent():
    from f3dasm._src.agentic.agents.strategizer import (
        STRATEGIZER_SYSTEM_PROMPT,
    )
    prompt = STRATEGIZER_SYSTEM_PROMPT
    # Must NOT make unconditional promises that agents are always present
    # Acceptable patterns: "when connected", "WHEN PRESENT", "if connected",
    # "when an OptimizationAgent is connected"
    # Forbidden: "always delegate to OptimizationAgent" (unconditional)
    # We check that the conditional qualifier appears near each agent name.
    # Coarse guard: if "WHEN PRESENT" or "when connected" appears, the
    # prompt is conditional — confirmed by test above.
    # Additionally, the prompt must NOT say the specialist "is always"
    # available.
    assert "is always available" not in prompt


def test_strategizer_owns_block1_doe():
    from f3dasm._src.agentic.agents.strategizer import (
        STRATEGIZER_SYSTEM_PROMPT,
    )
    # Strategizer must claim ownership of block 1 DoE
    prompt = STRATEGIZER_SYSTEM_PROMPT
    assert "You own block 1" in prompt or "own block 1" in prompt


# ---------------------------------------------------------------------------
# Graph smoke test — strategizer + implementer + OptimizationAgent + critic
# ---------------------------------------------------------------------------

def test_graph_with_optimization_agent_builds():
    """Graph containing OptimizationAgent constructs without error."""
    from f3dasm._src.agentic.backends.base import Agent, Edge, Graph
    from f3dasm._src.agentic.graph_builder import build_graph
    from f3dasm.agentic import OptimizationAgent

    class _Strat(Agent):
        role = "strategizer"
        description = "Stub strategizer."
        tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote"})

    class _Impl(Agent):
        role = "implementer"
        description = "Stub implementer."

    class _Critic(Agent):
        role = "critic"
        description = "Stub critic."

    spec = Graph(
        nodes={
            "s": _Strat(),
            "i": _Impl(),
            "opt": OptimizationAgent(),
            "c": _Critic(),
        },
        edges=(
            Edge("s", "i"),
            Edge("s", "opt"),
            Edge("s", "c"),
        ),
        entry="s",
    )

    class _StubAdapter:
        closure_tools: dict = {}
        def invoke(self, messages):
            return "## Report\nDone."

    graph = build_graph(
        spec, lambda name, agent: _StubAdapter()
    )
    assert hasattr(graph, "invoke")
