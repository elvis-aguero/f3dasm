"""Friction #5 fix: premature-Done becomes a 3-option soft nudge, and a new
CancelDelegation tool lets the strategizer detach a still-running delegation so
it stops blocking Done() (instead of bouncing on "wait for all delegations")."""
from __future__ import annotations

from f3dasm._src.agentic.backends.base import Agent, Edge, Graph
from f3dasm._src.agentic.nodes import StrategizerNode


class _Stub:
    def __init__(self) -> None:
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


def _node():
    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote",
                           "WriteDeliverable"})
        description = "strategizer"

    class B(Agent):
        role = "implementer"
        description = "implementer"

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),), entry="strategizer",
    )
    n = StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()},
    )
    return n


def test_cancel_delegation_detaches_and_unblocks():
    n = _node()
    n._registry["D001"] = {"status": "Working", "result": None, "evals": 0}
    out = n.adapter.closure_tools["CancelDelegation"]("D001")
    assert "cancelled" in out.lower() and "detached" in out.lower()
    assert n._registry["D001"]["status"] == "Cancelled"
    # premature-Done guard counts only Working → cancelled one no longer blocks
    pending = [d for d, e in n._registry.items() if e["status"] == "Working"]
    assert pending == []


def test_cancel_unknown_and_already_settled():
    n = _node()
    assert "no delegation" in n.adapter.closure_tools["CancelDelegation"](
        "D999").lower()
    n._registry["D002"] = {"status": "Done"}
    out = n.adapter.closure_tools["CancelDelegation"]("D002")
    assert "not" in out.lower() and "running" in out.lower()
    assert n._registry["D002"]["status"] == "Done"  # untouched


def test_premature_done_is_a_soft_three_option_nudge():
    n = _node()
    n._registry["D001"] = {"status": "Working"}
    out = n.adapter.closure_tools["Done"](summary="all done")
    # the nudge, not a hard error
    assert not out.lstrip().startswith("ERROR:")
    assert "three options" in out.lower()
    assert "CancelDelegation" in out and "GetStatus" in out
    # soft return → NOT counted as a tool error
    assert n._error_counts.get("strategizer", 0) == 0


def test_cancel_delegation_tool_is_registered():
    n = _node()
    assert "CancelDelegation" in n.adapter.closure_tools


def test_poll_escalation_offers_the_three_options():
    """Fix #5: a repeatedly-polled delegation gets the same 3 options as the
    premature-Done nudge (do other work / cancel / just wait), not just a
    'poll less' nag — so the agent never grinds out 30 status checks."""
    import time as _t
    n = _node()
    n._registry["D001"] = {
        "status": "Working", "result": None,
        "start_time": _t.monotonic(), "getstatus_count": 0}
    out = ""
    for _ in range(6):  # cross the >=5 escalation
        out = n.adapter.closure_tools["GetStatus"]("D001")
    assert out.lstrip().startswith("Working")
    assert "CancelDelegation('D001')" in out      # (b) cancel, real id
    assert "do other work" in out.lower()          # (a) do something else
    assert "just wait" in out.lower()              # (c) wait it out
    assert "wait=True" in out                       # future-proofing tip
