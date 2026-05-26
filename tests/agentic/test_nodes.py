"""Tests for AgentNode, StrategizerNode, ImplementerNode."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from langgraph.types import Command

from f3dasm._src.agentic.backends.base import Agent, Edge, Graph


# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------

class StubAdapter:
    """Minimal adapter stub for node tests."""

    def __init__(self, response: str = "## Done\nAll done.") -> None:
        self._response = response
        self.closure_tools: dict = {}

    def invoke(self, messages: list[dict]) -> str:
        return self._response


def _minimal_spec(name: str = "strategizer", target: str = "implementer") -> Graph:
    """Return a minimal two-node Graph for StrategizerNode tests."""
    class A(Agent):
        role = "strategizer"

    class B(Agent):
        pass

    return Graph(
        nodes={name: A(), target: B()},
        edges=(Edge(name, target),),
        entry=name,
    )


def make_state(
    messages=None,
    study_dir="/tmp",
    done=False,
    last_report=None,
    total_delegations=0,
    budget_seconds=None,
    return_to=None,
):
    from f3dasm._src.agentic.graph_state import AgenticState

    return AgenticState(
        messages=messages or [HumanMessage(content="Test problem")],
        study_dir=study_dir,
        done=done,
        last_report=last_report,
        total_delegations=total_delegations,
        budget_seconds=budget_seconds,
        return_to=return_to,
    )


# ---------------------------------------------------------------------------
# StrategizerNode tests
# ---------------------------------------------------------------------------


def test_strategizer_routes_done_when_done_called():
    """StrategizerNode returns Command(goto=END) when Done closure is called."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    class DoneCallingAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Done"](summary="Finished successfully.")
            return "Run complete."

    adapter = DoneCallingAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    cmd = node(make_state())

    assert cmd.goto == END
    assert cmd.update["done"] is True
    assert "Finished successfully." in (cmd.update.get("last_report") or "")


def test_strategizer_routes_delegate_when_delegate_called():
    """StrategizerNode returns Command(goto='implementer') when Delegate closure is called."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    class DelegateCallingAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](
                target="implementer",
                intent="Run experiment A.",
                expected_report="Report results.",
            )
            return "Task delegated. Waiting for Report."

    adapter = DelegateCallingAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    cmd = node(make_state())

    assert cmd.goto == "implementer"
    human_msgs = [m for m in cmd.update["messages"] if isinstance(m, HumanMessage)]
    assert any("Run experiment A." in m.content for m in human_msgs)
    assert cmd.update["total_delegations"] == 1
    assert cmd.update["return_to"] == "strategizer"


def test_done_blocked_while_delegation_pending():
    """Done returns an error when called after Delegate in the same turn."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    results: list[str] = []

    class DelegateThenDoneAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](
                target="implementer",
                intent="Run experiment.",
                expected_report="Report back.",
            )
            results.append(self.closure_tools["Done"](summary="All done."))
            return "Task delegated. Waiting for Report."

    adapter = DelegateThenDoneAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    cmd = node(make_state())

    # Done should have returned an error, not ended the run
    assert "implementer" in results[0]
    assert "not reported back" in results[0]
    # Routing should still be to the implementer
    assert cmd.goto == "implementer"


def test_strategizer_increments_delegation_count():
    """Each Delegate call increments total_delegations."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    class DelegateCallingAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](target="implementer", intent="task", expected_report="report")
            return "delegated"

    adapter = DelegateCallingAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    state = make_state(total_delegations=3)
    cmd = node(state)

    assert cmd.update["total_delegations"] == 4


def test_strategizer_no_routing_tool_defaults_to_done():
    """StrategizerNode ends run if no routing tool is called."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    adapter = StubAdapter(response="Final analysis complete.")
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    cmd = node(make_state())

    assert cmd.goto == END
    assert cmd.update["done"] is True


def test_strategizer_delegate_invalid_target_returns_error():
    """Delegate returns an error string for unknown targets."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    errors = []

    class BadTargetAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["Delegate"](
                target="nonexistent",
                intent="task",
                expected_report="report",
            )
            errors.append(result)
            return "done"

    adapter = BadTargetAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    node(make_state())

    assert errors and "ERROR" in errors[0]


# ---------------------------------------------------------------------------
# ImplementerNode tests
# ---------------------------------------------------------------------------


def test_implementer_returns_to_caller():
    """ImplementerNode routes back to return_to from state."""
    from f3dasm._src.agentic.nodes import ImplementerNode

    adapter = StubAdapter(response="## Report\nDone. Results: 42.")
    node = ImplementerNode(adapter)
    cmd = node(make_state(return_to="strategizer"))

    assert cmd.goto == "strategizer"
    assert "## Report" in cmd.update.get("last_report", "")


def test_implementer_stores_response_as_last_report():
    """ImplementerNode stores full response text in last_report."""
    from f3dasm._src.agentic.nodes import ImplementerNode

    response = "## Report\n### Actions taken\nRan code.\n### Conclusions\nResult: 3.14"
    adapter = StubAdapter(response=response)
    node = ImplementerNode(adapter)
    cmd = node(make_state(return_to="strategizer"))

    assert cmd.update["last_report"] == response


def test_implementer_accumulates_evals_when_report_evals_called():
    """ImplementerNode adds ReportEvals count to state evals_used."""
    from f3dasm._src.agentic.nodes import ImplementerNode

    class ReportEvalsCallingAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["ReportEvals"](count=1500)
            return "## Report\n### Actions taken\nDone.\n### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 1500"

    adapter = ReportEvalsCallingAdapter()
    node = ImplementerNode(adapter)
    state = make_state(return_to="strategizer")
    state["evals_used"] = 100
    cmd = node(state)

    assert cmd.update["evals_used"] == 1600


def test_implementer_evals_zero_when_report_evals_not_called():
    """ImplementerNode adds 0 to evals_used when ReportEvals is not called."""
    from f3dasm._src.agentic.nodes import ImplementerNode

    adapter = StubAdapter(response="## Report\n### Actions taken\nDone.\n### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0")
    node = ImplementerNode(adapter)
    state = make_state(return_to="strategizer")
    state["evals_used"] = 42
    cmd = node(state)

    assert cmd.update["evals_used"] == 42


def test_strategizer_delegate_includes_expected_report_in_message():
    """Delegate expected_report appears in the HumanMessage sent to implementer."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    class DelegateCallingAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](
                target="implementer",
                intent="Run the experiment.",
                expected_report="Must produce workspace/replicate.py",
            )
            return "Delegated."

    adapter = DelegateCallingAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    cmd = node(make_state())

    human_msgs = [m for m in cmd.update["messages"] if isinstance(m, HumanMessage)]
    full_content = " ".join(m.content for m in human_msgs)
    assert "workspace/replicate.py" in full_content
    assert "Required deliverables" in full_content


def test_strategizer_delegate_prepends_edge_preamble():
    """Edge preamble is prepended to the task message when the edge has one."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    class A(Agent):
        role = "strategizer"

    class B(Agent):
        pass

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer", preamble="PREAMBLE TEXT"),),
        entry="strategizer",
    )

    class DelegateCallingAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](
                target="implementer",
                intent="Do work.",
                expected_report="",
            )
            return "Delegated."

    adapter = DelegateCallingAdapter()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    cmd = node(make_state())

    human_msgs = [m for m in cmd.update["messages"] if isinstance(m, HumanMessage)]
    full_content = " ".join(m.content for m in human_msgs)
    assert "PREAMBLE TEXT" in full_content


# ---------------------------------------------------------------------------
# _to_adapter_messages tests
# ---------------------------------------------------------------------------


def test_to_adapter_messages_converts_human_and_ai():
    """_to_adapter_messages converts HumanMessage and AIMessage correctly."""
    from f3dasm._src.agentic.nodes import _to_adapter_messages

    msgs = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
        HumanMessage(content="Bye"),
    ]
    result = _to_adapter_messages(msgs)

    assert len(result) == 3
    assert result[0] == {"role": "user", "content": "Hello"}
    assert result[1] == {"role": "ai", "content": "Hi there"}
    assert result[2] == {"role": "user", "content": "Bye"}
