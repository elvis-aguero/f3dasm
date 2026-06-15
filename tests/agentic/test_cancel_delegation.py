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


def _seed_store(store_dir, delegation_id):
    from f3dasm._src.agentic.instrumented import InstrumentedDataGenerator
    from f3dasm._src.core import DataGenerator
    from f3dasm._src.experimentsample import ExperimentSample, JobStatus

    class _Sum(DataGenerator):
        def execute(self, s, **k):
            s._output_data["f"] = sum(s._input_data.values())
            s.job_status = JobStatus.FINISHED
            return s

    gen = InstrumentedDataGenerator(
        inner=_Sum(), store_dir=store_dir, delegation_id=delegation_id,
        flush_every=1)
    gen.execute(ExperimentSample(
        _input_data={"x0": 1.0}, _output_data={}, job_status=JobStatus.OPEN))
    gen.flush()


def test_cancel_is_two_shot_for_progressing_delegation(tmp_path):
    """A delegation already writing ledgered evals is progressing, not stuck:
    the first cancel is HELD, a deliberate second call confirms."""
    run_dir = tmp_path / "runs" / "T0"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    _seed_store(run_dir / "experiment_data", "D004")
    n = _node()
    n._current_notes_dir = run_dir / "debug" / "strategizer_notes"
    n._registry["D004"] = {"status": "Working", "result": None, "evals": 0}

    out1 = n.adapter.closure_tools["CancelDelegation"]("D004")
    assert "HOLD" in out1
    assert n._registry["D004"]["status"] == "Working"   # NOT cancelled yet
    out2 = n.adapter.closure_tools["CancelDelegation"]("D004")
    assert "cancelled" in out2.lower()
    assert n._registry["D004"]["status"] == "Cancelled"


def test_cancel_single_shot_when_no_ledgered_evals(tmp_path):
    """No ledgered evals → cancel is immediate (the impatience guard only
    trips for delegations actually producing evaluations)."""
    run_dir = tmp_path / "runs" / "T1"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    (run_dir / "experiment_data").mkdir()
    n = _node()
    n._current_notes_dir = run_dir / "debug" / "strategizer_notes"
    n._registry["D003"] = {"status": "Working", "result": None, "evals": 0}
    out = n.adapter.closure_tools["CancelDelegation"]("D003")
    assert n._registry["D003"]["status"] == "Cancelled"
    assert "HOLD" not in out
