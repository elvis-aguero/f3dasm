"""Tests for route-aware run termination.

Covers:
- Bounded re-prompt on unaccepted termination (max 3 attempts)
- UNGATED banner on exhausted attempts
- Accepted Done() → clean END with no banner
- Run-level cost backstop (elapsed > RUN_BACKSTOP_MULTIPLE * budget)
- Working delegations survive loopbacks
- Ledger duplicate-statement guard
- ReadNote directory guard
"""
from __future__ import annotations

import tempfile
import time
import threading
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from langgraph.types import Command

from f3dasm._src.agentic.backends.base import Agent, Edge, Graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StubAdapter:
    """Minimal adapter stub."""

    def __init__(self, response: str = "Final analysis complete.") -> None:
        self._response = response
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.invoke_count = 0

    def invoke(self, messages: list[dict]) -> str:
        self.invoke_count += 1
        return self._response


def _minimal_spec(name: str = "strategizer", target: str = "implementer") -> Graph:
    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote", "WriteDeliverable"})
        description = "Test strategizer."

    class B(Agent):
        description = "Test implementer."

    return Graph(
        nodes={name: A(), target: B()},
        edges=(Edge(name, target),),
        entry=name,
    )


def _make_state(study_dir=None, messages=None, **kwargs):
    from f3dasm._src.agentic.graph_state import AgenticState

    if study_dir is None:
        d = Path(tempfile.mkdtemp(prefix="f3dasm_rat_"))
        (d / "replicate.py").write_text("# test replicate\n")
        study_dir = d
    return AgenticState(
        messages=messages or [HumanMessage(content="Test problem")],
        study_dir=str(study_dir),
        done=False,
        last_report=None,
        total_delegations=0,
        budget_seconds=kwargs.pop("budget_seconds", None),
        return_to=kwargs.pop("return_to", None),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Bounded re-prompt on unaccepted termination
# ---------------------------------------------------------------------------


def test_unaccepted_termination_reprompts():
    """When adapter doesn't call Done(), node loops back with diagnostic message."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    # Adapter never calls Done
    adapter = StubAdapter(response="Final analysis complete.")
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
    )

    # Create study_dir with replicate.py so missing-deliverables is NOT the issue
    study_dir = Path(tempfile.mkdtemp(prefix="f3dasm_rat_"))
    (study_dir / "replicate.py").write_text("# test\n")
    state = _make_state(study_dir=study_dir)

    cmd = node(state)

    # Should loop back, not terminate
    assert cmd.goto == "strategizer", f"Expected loopback, got goto={cmd.goto!r}"
    assert node._finish_attempts == 1

    # Injected message should explain the issue
    messages = cmd.update.get("messages", [])
    human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    assert human_msgs, "No HumanMessage injected in loopback update"
    assert any(
        "without an accepted Done" in m.content for m in human_msgs
    ), f"Expected 'without an accepted Done' in messages; got: {[m.content for m in human_msgs]}"


def test_ungated_finish_after_three_attempts():
    """After 3 loopbacks, 4th call terminates with UNGATED banner."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    # Adapter never calls Done; replicate.py present so only Done is missing
    study_dir = Path(tempfile.mkdtemp(prefix="f3dasm_rat_"))
    (study_dir / "replicate.py").write_text("# test\n")

    adapter = StubAdapter(response="Final analysis complete.")
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
    )

    # Simulate 3 loopbacks by calling __call__ and re-feeding messages
    state = _make_state(study_dir=study_dir)
    for attempt in range(1, 4):
        cmd = node(state)
        assert cmd.goto == "strategizer", (
            f"Attempt {attempt}: expected loopback, got goto={cmd.goto!r}"
        )
        assert node._finish_attempts == attempt
        # Feed the injected messages back into state for next call
        state = _make_state(
            study_dir=study_dir,
            messages=list(state["messages"]) + list(cmd.update.get("messages", [])),
        )

    # 4th call — attempts exhausted — must terminate
    cmd = node(state)
    assert cmd.goto == END, f"Expected END after 3 attempts, got goto={cmd.goto!r}"
    assert cmd.update.get("done") is True
    last_report = cmd.update.get("last_report", "")
    assert "UNGATED RUN" in last_report, (
        f"Expected 'UNGATED RUN' in last_report; got: {last_report!r}"
    )


def test_accepted_done_no_banner():
    """Full Done() dance → END with no UNGATED banner; _finish_attempts stays 0."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    study_dir = Path(tempfile.mkdtemp(prefix="f3dasm_rat_"))
    (study_dir / "replicate.py").write_text("# test\n")

    class DoneCallingAdapter(StubAdapter):
        def invoke(self, messages):
            self.invoke_count += 1
            self.closure_tools["Done"](summary="All done.")  # first: warning
            self.closure_tools["Done"](summary="All done.")  # second: accepted
            return "Run complete."

    adapter = DoneCallingAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
    )
    state = _make_state(study_dir=study_dir)
    cmd = node(state)

    assert cmd.goto == END, f"Expected END, got {cmd.goto!r}"
    assert cmd.update.get("done") is True
    last_report = cmd.update.get("last_report", "")
    assert "UNGATED" not in last_report, (
        f"Expected no UNGATED banner in accepted run; got: {last_report!r}"
    )
    assert node._finish_attempts == 0


# ---------------------------------------------------------------------------
# 2. Hard budget stop
# ---------------------------------------------------------------------------


def test_run_backstop_aborts_past_multiple():
    """Past RUN_BACKSTOP_MULTIPLE x budget, invoke is skipped; RUN BACKSTOP."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    study_dir = Path(tempfile.mkdtemp(prefix="f3dasm_rat_"))
    (study_dir / "replicate.py").write_text("# test\n")

    adapter = StubAdapter(response="Should not be called.")
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
    )

    # budget=10s, started 100s ago → elapsed = 10x budget >> 2x backstop
    state = _make_state(study_dir=study_dir)
    state["budget_seconds"] = 10
    state["start_time"] = time.time() - 100

    cmd = node(state)

    assert adapter.invoke_count == 0, (
        f"adapter.invoke called {adapter.invoke_count}x; should be skipped"
    )
    assert cmd.goto == END
    assert cmd.update.get("done") is True
    assert "RUN BACKSTOP" in cmd.update.get("last_report", "")


def test_soft_budget_does_not_terminate_below_backstop():
    """Time budget is SOFT: past 100% but below the backstop, the run
    CONTINUES (adapter.invoke is called) — warning only, no force-end."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    study_dir = Path(tempfile.mkdtemp(prefix="f3dasm_rat_"))
    (study_dir / "replicate.py").write_text("# test\n")

    adapter = StubAdapter(response="Continuing despite soft warning.")
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
    )

    # budget=10s, started 15s ago → 1.5x budget: over 100%, under 2x
    state = _make_state(study_dir=study_dir)
    state["budget_seconds"] = 10
    state["start_time"] = time.time() - 15

    node(state)

    assert adapter.invoke_count == 1, (
        "soft budget must NOT force-terminate below the backstop"
    )


# ---------------------------------------------------------------------------
# 3. Working delegations survive loopback
# ---------------------------------------------------------------------------


def test_working_delegations_survive_loopback():
    """A Working registry entry is not cleared by the A1/A2 reset on loopback."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    study_dir = Path(tempfile.mkdtemp(prefix="f3dasm_rat_"))
    (study_dir / "replicate.py").write_text("# test\n")

    # Adapter does NOT call Done → triggers loopback
    adapter = StubAdapter(response="Still thinking.")
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
    )

    # Seed the registry with a Working entry BEFORE __call__
    fake_event = threading.Event()
    with node._registry_lock:
        node._registry["D001"] = {
            "status": "Working",
            "result": None,
            "evals": 0,
            "hypothesis_ids": [],
            "started_at": "2026-01-01T00:00:00+00:00",
            "start_time": time.monotonic(),
            "followup_event": fake_event,
            "followup_question": None,
            "followup_answer": None,
            "followup_count": 0,
        }
        node._threads["D001"] = threading.Thread(target=lambda: None)

    state = _make_state(study_dir=study_dir)
    cmd = node(state)

    # Should be a loopback
    assert cmd.goto == "strategizer", f"Expected loopback, got {cmd.goto!r}"

    # The Working entry must still be in the registry
    with node._registry_lock:
        assert "D001" in node._registry, (
            f"D001 was cleared from registry during loopback; registry={node._registry}"
        )
        assert node._registry["D001"]["status"] == "Working"


# ---------------------------------------------------------------------------
# 4. Ledger duplicate-statement guard
# ---------------------------------------------------------------------------


def test_propose_rejects_duplicate_statement(tmp_path):
    """Proposing an identical statement twice returns ERROR citing the first H-id."""
    from f3dasm._src.agentic.hypothesis_ledger import HypothesisLedger

    ledger = HypothesisLedger(tmp_path)

    stmt = "Optimal t/L lies near 0.08 for re-entrant geometry"
    h1 = ledger.propose(
        statement=stmt,
        falsification_criterion="A design outside [0.07,0.09] beats buckling_load 1.47",
        prediction="Best design found in range 0.07-0.09",
        prior=0.6,
        proposed_by="strategizer",
    )
    assert h1 == "H1", f"First propose should return H1, got {h1!r}"

    # Exact duplicate
    result = ledger.propose(
        statement=stmt,
        falsification_criterion="Different criterion",
        prediction="Different prediction",
        prior=0.4,
        proposed_by="strategizer",
    )
    assert result.startswith("ERROR"), f"Expected ERROR for duplicate, got {result!r}"
    assert "H1" in result, f"Expected H1 in error message, got {result!r}"
    assert "duplicate" in result.lower(), f"Expected 'duplicate' in error, got {result!r}"


def test_propose_rejects_duplicate_case_whitespace(tmp_path):
    """Duplicate detection is case- and whitespace-insensitive."""
    from f3dasm._src.agentic.hypothesis_ledger import HypothesisLedger

    ledger = HypothesisLedger(tmp_path)

    stmt = "Optimal t/L lies near 0.08 for re-entrant geometry"
    h1 = ledger.propose(
        statement=stmt,
        falsification_criterion="Counter-example outside range",
        prediction="Best in-range",
        prior=0.6,
        proposed_by="strategizer",
    )
    assert h1 == "H1"

    # Variant with different case and extra whitespace
    variant = "  OPTIMAL  T/L  LIES  NEAR  0.08  FOR  RE-ENTRANT  GEOMETRY  "
    result = ledger.propose(
        statement=variant,
        falsification_criterion="Counter",
        prediction="Best",
        prior=0.4,
        proposed_by="strategizer",
    )
    assert result.startswith("ERROR"), f"Expected ERROR for case/whitespace variant, got {result!r}"
    assert "H1" in result


# ---------------------------------------------------------------------------
# 5. ReadNote directory guard
# ---------------------------------------------------------------------------


def test_readnote_directory_returns_error(tmp_path):
    """ReadNote on a directory path returns an error string instead of raising."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    (tmp_path / "replicate.py").write_text("# test\n")
    # Create a subdirectory to pass as the ReadNote path
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    results: list[str] = []

    class ReadNoteAdapter(StubAdapter):
        def invoke(self, messages):
            self.invoke_count += 1
            result = self.closure_tools["ReadNote"]("subdir")  # points to a directory
            results.append(result)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = ReadNoteAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        study_dir=str(tmp_path),
    )
    state = _make_state(study_dir=tmp_path)
    cmd = node(state)

    assert results, "ReadNote was never called"
    assert "ERROR" in results[0], (
        f"Expected ERROR from ReadNote on directory, got: {results[0]!r}"
    )
    assert "directory" in results[0].lower(), (
        f"Expected 'directory' in error, got: {results[0]!r}"
    )
