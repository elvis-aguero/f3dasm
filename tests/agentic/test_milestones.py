"""Milestone ledger (Spec C2): process policy, soft-enforced, auto-satisfying."""
from __future__ import annotations

from f3dasm._src.agentic.backends.base import Agent, Edge, Graph
from f3dasm._src.agentic.delegation_log import DelegationLog
from f3dasm._src.agentic.milestones import (
    MilestoneLedger,
    blocking_gate,
    milestone_gate_nudge,
)
from f3dasm._src.agentic.nodes import StrategizerNode


class _Stub:
    def __init__(self):
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


# --------------------------------------------------------------------------
# Ledger unit behaviour
# --------------------------------------------------------------------------

def test_seed_defaults_creates_pending_gates(tmp_path):
    led = MilestoneLedger(tmp_path)
    led.seed_defaults()
    keys = {m["key"] for m in led.list_all()}
    assert {"lit_review_before_doe", "datagenerator_gold_state"} <= keys
    assert all(m["status"] == "PENDING" and m["gate"] for m in led.list_all())


def test_seed_is_idempotent(tmp_path):
    led = MilestoneLedger(tmp_path)
    led.seed_defaults()
    led.seed_defaults()
    assert len(led.list_all()) == 2  # not duplicated


def test_seed_respects_disabled(tmp_path):
    led = MilestoneLedger(tmp_path)
    led.seed_defaults(disabled=frozenset({"lit_review_before_doe"}))
    keys = {m["key"] for m in led.list_all()}
    assert "lit_review_before_doe" not in keys
    assert "datagenerator_gold_state" in keys


def test_propose_complete_skip(tmp_path):
    led = MilestoneLedger(tmp_path)
    mid = led.propose("draft the candidate pipeline", phase="optimization")
    assert mid.startswith("M")
    assert led.get(mid)["source"] == "agent"
    led.complete(mid, "done it")
    assert led.get(mid)["status"] == "DONE"
    mid2 = led.propose("optional thing")
    led.skip(mid2, "not needed here")
    assert led.get(mid2)["status"] == "SKIPPED"


def test_pending_gates_filtered_by_phase(tmp_path):
    led = MilestoneLedger(tmp_path)
    led.seed_defaults()
    doe = led.pending_gates("doe")
    assert len(doe) == 1 and doe[0]["key"] == "lit_review_before_doe"
    assert led.pending_gates("ml") == []


# --------------------------------------------------------------------------
# Auto-satisfy + gate nudge against a real strategizer node
# --------------------------------------------------------------------------

def _node(tmp_path):
    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done"})
        description = "strategizer"

    class Lit(Agent):
        role = "literature_reviewer"
        description = "lit"

    class B(Agent):
        role = "implementer"
        description = "impl"

    spec = Graph(
        nodes={"strategizer": A(), "literature_reviewer": Lit(),
               "implementer": B()},
        edges=(Edge("strategizer", "literature_reviewer"),
               Edge("strategizer", "implementer")), entry="strategizer")
    notes = tmp_path / "debug" / "strategizer_notes"
    notes.mkdir(parents=True)
    dlog = DelegationLog(tmp_path / "debug" / "delegation_log.jsonl")
    n = StrategizerNode(
        _Stub(), name="strategizer",
        outgoing=["literature_reviewer", "implementer"], spec=spec,
        worker_adapters={"literature_reviewer": _Stub(), "implementer": _Stub()},
        notes_dir=notes, delegation_log=dlog)
    return n


def test_gate_nudges_then_auto_satisfies_after_lit_review(tmp_path):
    n = _node(tmp_path)
    led = n._milestones
    # entering DoE with no lit review yet → nudge fires
    nudge = milestone_gate_nudge(led, n, "doe")
    assert "MILESTONE CHECKPOINT" in nudge
    assert "literature review" in nudge.lower()
    # a completed literature_reviewer delegation satisfies the predicate
    n._delegation_log.record(
        id="D001", from_node="strategizer", to_node="literature_reviewer",
        task="survey", deliverable="done", hypothesis_ids=[],
        started_at="t0", completed_at="t1", status="DONE")
    nudge2 = milestone_gate_nudge(led, n, "doe")
    assert nudge2 == ""  # auto-satisfied → no nag
    assert led.get([m["id"] for m in led.list_all()
                    if m["key"] == "lit_review_before_doe"][0])["status"] == "DONE"


def test_milestone_closures_registered_on_strategizer(tmp_path):
    n = _node(tmp_path)
    for tool in ("MilestoneList", "MilestonePropose", "MilestoneComplete",
                 "MilestoneSkip"):
        assert tool in n.adapter.closure_tools


def test_blocking_gate_logic(tmp_path):
    """A pending gate blocks at-or-after its phase; literature/setup/None exempt."""
    n = _node(tmp_path)
    led = n._milestones
    # lit_review gates 'doe' → blocks doe AND later phases
    assert blocking_gate(led, n, "doe") is not None
    assert blocking_gate(led, n, "ml") is not None
    # exempt: untagged, and the phases used to satisfy gates
    assert blocking_gate(led, n, None) is None
    assert blocking_gate(led, n, "literature") is None
    assert blocking_gate(led, n, "setup") is None


def test_delegate_into_gated_phase_is_hard_blocked(tmp_path):
    """Fix #1: delegating gated-phase work while a gate is pending is REFUSED
    (the delegation does not fire), not merely nudged."""
    n = _node(tmp_path)
    hid = n.adapter.closure_tools["HypothesisPropose"](
        "stmt", "crit", "pred", 0.5)
    before = len(n._registry)
    out = n.adapter.closure_tools["Delegate"](
        "implementer", "generate data", "report", hypothesis_ids=[hid],
        wait=True, phase="data_generation")
    assert out.startswith("BLOCKED")
    assert "MilestoneSkip" in out
    assert len(n._registry) == before  # nothing fired


def test_milestone_skip_unblocks_delegation(tmp_path):
    """The unilateral escape: skipping the gate (with a reason) lets the
    delegation through — no deadlock."""
    n = _node(tmp_path)
    hid = n.adapter.closure_tools["HypothesisPropose"](
        "stmt", "crit", "pred", 0.5)
    # skip both gates that would block data_generation
    for m in n._milestones.list_all():
        n.adapter.closure_tools["MilestoneSkip"](m["id"], "not needed in test")
    out = n.adapter.closure_tools["Delegate"](
        "implementer", "generate data", "report", hypothesis_ids=[hid],
        wait=True, phase="data_generation")
    assert not out.startswith("BLOCKED")
    assert out.lstrip().startswith(("Done", "Errored"))


def test_milestone_complete_requires_a_brief(tmp_path):
    """Fix #1: ticking needs a one-line why — no rubber-stamping."""
    n = _node(tmp_path)
    mid = n._milestones.propose("my own step")
    assert "ERROR" in n.adapter.closure_tools["MilestoneComplete"](mid, "")
    assert "ERROR" in n.adapter.closure_tools["MilestoneComplete"](mid, "   ")
    ok = n.adapter.closure_tools["MilestoneComplete"](mid, "did the thing via D2")
    assert "ERROR" not in ok
    assert n._milestones.get(mid)["status"] == "DONE"


def test_done_blocked_until_milestones_resolved(tmp_path):
    """Fix #1 (bare minimum): Done() cannot close while any milestone is
    PENDING; resolving (skip/complete) all of them unblocks it."""
    n = _node(tmp_path)
    out = n.adapter.closure_tools["Done"](summary="done")
    assert "Cannot close yet" in out and "PENDING" in out
    # resolve every milestone, then Done() proceeds past the close-gate
    for m in n._milestones.list_all():
        n.adapter.closure_tools["MilestoneSkip"](m["id"], "n/a for test")
    out2 = n.adapter.closure_tools["Done"](summary="done")
    assert "Cannot close yet" not in out2
