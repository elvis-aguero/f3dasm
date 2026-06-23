"""A milestone-blocked delegation must NOT burn a delegation ID.

next_id() advances a monotonic counter on every call; it used to be called at
the top of Delegate(), before the milestone gate, so a blocked implementer
attempt consumed an ID that never reached the log — leaving a permanent gap in
the sequence (the always-present early MILESTONE_BLOCK reliably ate "D002").
The fix moves allocation to AFTER the gate; this pins it.
"""
from __future__ import annotations

from f3dasm._src.agentic.backends.base import Agent, Edge, Graph
from f3dasm._src.agentic.delegation_log import DelegationLog
from f3dasm._src.agentic.nodes import StrategizerNode


class _Stub:
    def __init__(self):
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


def _node(tmp_path):
    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done"})
        description = "strategizer"

    class B(Agent):
        role = "implementer"
        description = "implementer"

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),), entry="strategizer")
    notes = tmp_path / "debug" / "strategizer_notes"
    notes.mkdir(parents=True)
    return StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()}, notes_dir=notes,
        delegation_log=DelegationLog(tmp_path / "dlog.jsonl"))


def test_milestone_blocked_delegation_does_not_consume_an_id(tmp_path):
    n = _node(tmp_path)
    # Seed a pending milestone → the implementer is gated until it is resolved.
    n._milestones.propose("resolve the process backlog first")
    assert n._milestones.pending(), "precondition: a milestone should be pending"

    # First implementer attempt is BLOCKED (must not record or allocate an ID).
    blocked = n.adapter.closure_tools["Delegate"](
        "implementer", "run a sweep", "a report", wait=True)
    assert blocked.lstrip().startswith("BLOCKED"), blocked

    # The shared ID counter must NOT have advanced: the next allocation is still
    # the first ID. (With the bug — allocation before the gate — the blocked
    # attempt would have consumed D001, so this would return "D002".)
    assert n._delegation_log.next_id() == "D001", "blocked attempt burned an ID"
