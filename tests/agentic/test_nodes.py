"""Tests for AgentNode, StrategizerNode, ImplementerNode."""
from __future__ import annotations

import threading
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


def test_strategizer_delegate_returns_task_id():
    """Delegate() returns a TASK-xxxxxxxx ID and starts a background thread."""
    import threading
    import time
    from f3dasm._src.agentic.nodes import StrategizerNode

    received_ids: list[str] = []
    worker_started = threading.Event()

    class SlowWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            worker_started.set()
            time.sleep(0.05)
            return "## Report\n### Actions taken\nDone.\n### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"

    class DelegateCallingAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["Delegate"](
                target="implementer",
                intent="Run experiment A.",
                expected_report="Report results.",
            )
            received_ids.append(result)
            # Wait briefly then call Done after delegation finishes
            worker_started.wait(timeout=2)
            time.sleep(0.1)  # let worker finish
            self.closure_tools["Done"](summary="All done.")
            return "Done."

    adapter = DelegateCallingAdapter()
    spec = _minimal_spec()
    worker = SlowWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    cmd = node(make_state())

    assert cmd.goto == END
    assert received_ids and "TASK-" in received_ids[0]


def test_done_blocked_while_delegation_pending():
    """Done returns an error when called while a delegation is still Working."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    results: list[str] = []

    class SlowWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            import time
            time.sleep(10)  # never finishes in test timeframe
            return "## Report\n### Actions taken\nDone.\n### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"

    class DelegateThenDoneAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](
                target="implementer",
                intent="Run experiment.",
                expected_report="Report back.",
            )
            results.append(self.closure_tools["Done"](summary="All done."))
            # Fall through to Done without waiting → should default to END
            self.closure_tools["Done"](summary="forced")
            return "Done."

    adapter = DelegateThenDoneAdapter()
    spec = _minimal_spec()
    worker = SlowWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    node(make_state())

    # Done should have returned an error about pending delegation
    assert results and "ERROR" in results[0]
    assert "still running" in results[0]


def test_strategizer_increments_delegation_count():
    """total_delegations reflects all fired delegations after the run completes."""
    import time
    from f3dasm._src.agentic.nodes import StrategizerNode

    class FastWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            return "## Report\n### Actions taken\nDone.\n### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"

    class DelegateCallingAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](target="implementer", intent="task", expected_report="report")
            time.sleep(0.1)  # let worker finish
            self.closure_tools["Done"](summary="done")
            return "Done."

    adapter = DelegateCallingAdapter()
    spec = _minimal_spec()
    worker = FastWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    state = make_state(total_delegations=3)
    cmd = node(state)

    # 3 existing + 1 new delegation from registry
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
    """Delegate expected_report appears in the task message sent to the worker."""
    import time
    from f3dasm._src.agentic.nodes import StrategizerNode

    received_messages: list[list] = []

    class CapturingWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            received_messages.append(messages)
            return "## Report\n### Actions taken\nDone.\n### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"

    class DelegateCallingAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](
                target="implementer",
                intent="Run the experiment.",
                expected_report="Must produce workspace/replicate.py",
            )
            time.sleep(0.1)  # let worker finish
            self.closure_tools["Done"](summary="done")
            return "Done."

    adapter = DelegateCallingAdapter()
    spec = _minimal_spec()
    worker = CapturingWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    node(make_state())

    assert received_messages, "Worker was never called"
    task_content = received_messages[0][0]["content"]
    assert "workspace/replicate.py" in task_content
    assert "Required deliverables" in task_content


def test_strategizer_delegate_prepends_edge_preamble():
    """Edge preamble is prepended to the task message when the edge has one."""
    import time
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

    received_messages: list[list] = []

    class CapturingWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            received_messages.append(messages)
            return "## Report\n### Actions taken\nDone.\n### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"

    class DelegateCallingAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](
                target="implementer",
                intent="Do work.",
                expected_report="",
            )
            time.sleep(0.1)  # let worker finish
            self.closure_tools["Done"](summary="done")
            return "Done."

    adapter = DelegateCallingAdapter()
    worker = CapturingWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    node(make_state())

    assert received_messages, "Worker was never called"
    task_content = received_messages[0][0]["content"]
    assert "PREAMBLE TEXT" in task_content


# ---------------------------------------------------------------------------
# Parallel fan-out tests
# ---------------------------------------------------------------------------


def test_parallel_two_delegations_both_complete():
    """Strategizer can fire two delegations concurrently; both complete and are counted."""
    import time
    from f3dasm._src.agentic.nodes import StrategizerNode

    call_log: list[str] = []

    class LoggingWorkerAdapter(StubAdapter):
        def __init__(self, name: str) -> None:
            super().__init__()
            self._name = name

        def invoke(self, messages: list) -> str:
            call_log.append(self._name)
            time.sleep(0.05)
            return (
                f"## Report\n### Actions taken\nDone {self._name}.\n"
                f"### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"
            )

    class TwoDelegateAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](target="worker_a", intent="Task A", expected_report="")
            self.closure_tools["Delegate"](target="worker_b", intent="Task B", expected_report="")
            time.sleep(0.15)  # let both workers finish
            result = self.closure_tools["Done"](summary="Both done.")
            assert "ERROR" not in result, f"Done() failed: {result}"
            return "Done."

    class A(Agent):
        role = "strategizer"

    class B(Agent):
        pass

    spec = Graph(
        nodes={"strategizer": A(), "worker_a": B(), "worker_b": B()},
        edges=(Edge("strategizer", "worker_a"), Edge("strategizer", "worker_b")),
        entry="strategizer",
    )

    adapter = TwoDelegateAdapter()
    node = StrategizerNode(
        adapter, name="strategizer",
        outgoing=["worker_a", "worker_b"], spec=spec,
        worker_adapters={
            "worker_a": LoggingWorkerAdapter("worker_a"),
            "worker_b": LoggingWorkerAdapter("worker_b"),
        },
    )
    cmd = node(make_state())

    assert cmd.goto == END
    assert cmd.update["done"] is True
    assert cmd.update["total_delegations"] == 2  # 0 existing + 2 from registry
    assert set(call_log) == {"worker_a", "worker_b"}


def test_get_status_returns_working_then_done():
    """GetStatus returns 'Working' while the delegation is running, then 'Done'."""
    import time
    from f3dasm._src.agentic.nodes import StrategizerNode

    status_snapshots: list[str] = []

    class SlowWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            time.sleep(0.1)
            return (
                "## Report\n### Actions taken\nDone.\n"
                "### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"
            )

    class PollAdapter(StubAdapter):
        def invoke(self, messages):
            task_id_msg = self.closure_tools["Delegate"](
                target="implementer", intent="Work.", expected_report=""
            )
            # Extract TASK-xxxxxxxx from the return string
            import re
            m = re.search(r"TASK-[0-9a-f]+", task_id_msg)
            assert m, f"No task ID found in: {task_id_msg!r}"
            task_id = m.group()

            # Poll immediately — should be Working
            status_snapshots.append(self.closure_tools["GetStatus"](task_id))
            # Wait and poll again — should be Done
            time.sleep(0.5)
            status_snapshots.append(self.closure_tools["GetStatus"](task_id))
            self.closure_tools["Done"](summary="polled")
            return "Done."

    adapter = PollAdapter()
    spec = _minimal_spec()
    worker = SlowWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    node(make_state())

    assert status_snapshots[0] == "Working"  # no notification yet
    assert "Done" in status_snapshots[1]      # notification prefix + Done report


def test_registry_cleared_between_runs():
    """Registry and ask_count are reset on each __call__, preventing cross-run pollution."""
    import time
    from f3dasm._src.agentic.nodes import StrategizerNode

    call_count = [0]

    class FastWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            return "## Report\n### Actions taken\nDone.\n### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"

    class OneDelegateAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](target="implementer", intent="task", expected_report="")
            time.sleep(0.1)
            self.closure_tools["Done"](summary="done")
            call_count[0] += 1
            return "Done."

    adapter = OneDelegateAdapter()
    spec = _minimal_spec()
    worker = FastWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )

    # First run
    cmd1 = node(make_state(total_delegations=0))
    assert cmd1.update["total_delegations"] == 1

    # Second run on the same node — registry must be fresh
    cmd2 = node(make_state(total_delegations=0))
    assert cmd2.update["total_delegations"] == 1  # not 2


def test_get_status_unknown_id_returns_error():
    """GetStatus on an unknown ID returns a clear ERROR string."""
    from f3dasm._src.agentic.nodes import StrategizerNode

    errors: list[str] = []

    class PollUnknownAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["GetStatus"]("TASK-notreal")
            errors.append(result)
            self.closure_tools["Done"](summary="done")
            return "Done."

    adapter = PollUnknownAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    node(make_state())

    assert errors and errors[0].startswith("ERROR")
    assert "TASK-notreal" in errors[0]


def test_errored_status_contains_traceback():
    """When a worker raises, GetStatus returns 'Errored:\\n<traceback>' with enough info."""
    import re
    import time
    from f3dasm._src.agentic.nodes import StrategizerNode

    status_seen: list[str] = []
    worker_done = threading.Event()

    class CrashingWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            raise ValueError("pool.csv not found at workspace/pool.csv")

    class PollErrorAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["Delegate"](
                target="implementer", intent="task", expected_report=""
            )
            import re as _re
            task_id = _re.search(r"TASK-[0-9a-f]+", result).group()
            # Poll until resolved
            for _ in range(50):
                status = self.closure_tools["GetStatus"](task_id)
                if not status == "Working":
                    status_seen.append(status)
                    break
                time.sleep(0.02)
            self.closure_tools["Done"](summary="done")
            return "Done."

    adapter = PollErrorAdapter()
    spec = _minimal_spec()
    worker = CrashingWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    node(make_state())

    assert status_seen, "Never got a non-Working status"
    msg = status_seen[0]
    assert "Errored:" in msg  # may have notification prefix
    # Must contain the exception type and message for diagnostics (B1)
    assert "ValueError" in msg
    assert "pool.csv" in msg


def test_delegation_timeout_marks_errored():
    """A hung delegation is marked Errored after the timeout elapses."""
    import time
    from f3dasm._src.agentic.nodes import StrategizerNode

    status_seen: list[str] = []
    worker_started = threading.Event()

    class HungWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            worker_started.set()
            time.sleep(60)  # hangs
            return "never"

    class TimeoutPollAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["Delegate"](
                target="implementer", intent="task", expected_report=""
            )
            import re as _re
            task_id = _re.search(r"TASK-[0-9a-f]+", result).group()
            worker_started.wait(timeout=2)
            time.sleep(0.05)  # let the timeout expire
            status = self.closure_tools["GetStatus"](task_id)
            status_seen.append(status)
            # Force Done with the errored delegation still in registry
            # (timeout marks it Errored so Done() should now succeed)
            self.closure_tools["Done"](summary="timed out")
            return "Done."

    adapter = TimeoutPollAdapter()
    spec = _minimal_spec()
    worker = HungWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    # budget_seconds=0.013 → delegation timeout = 75% * 0.013 ≈ 10ms
    state = make_state()
    state["budget_seconds"] = 0.013
    state["start_time"] = 0.0  # unused but needed for budget warning path

    cmd = node(state)
    assert cmd.goto == END
    assert status_seen and "Timeout" in status_seen[0]


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
