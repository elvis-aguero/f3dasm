"""LangGraph node classes for the f3dasm agentic runtime.

Each node class implements ``__call__(state: AgenticState) -> Command``,
which is the ADAS-inspectable topology entry point.
``inspect.getsource(StrategizerNode.__call__)`` reads the full routing logic.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph_state import AgenticState

__all__ = [
    "AgentNode", "StrategizerNode", "ImplementerNode", "_to_adapter_messages"
]

_REQUIRED_SUBSECTIONS = [
    "### Actions taken",
    "### Files touched",
    "### Conclusions",
    "### Numbers",
]
_CAPABILITY_PHRASES = [
    "i cannot", "i can't", "i don't have access",
    "unable to", "not able to", "i am unable",
]


def _to_adapter_messages(lc_messages: list) -> list[dict]:
    """Convert LangChain message objects to adapter-format dicts."""
    from langchain_core.messages import AIMessage, HumanMessage

    result: list[dict] = []
    for msg in lc_messages:
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            result.append({"role": "user", "content": str(content)})
        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            result.append({"role": "ai", "content": str(content)})
    return result


def _classify_response(text: str) -> str | None:
    """Return a REFLECT diagnosis string if text is malformed, else None."""
    from .agent_prompts import (
        REFLECT_DIAGNOSIS_CAPABILITY_LIMIT,
        REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE,
        REFLECT_DIAGNOSIS_NO_REPORT_HEADING,
        REFLECT_DIAGNOSIS_SHORT,
    )

    if len(text.strip()) < 100:
        return REFLECT_DIAGNOSIS_SHORT
    low = text.lower()
    if any(p in low for p in _CAPABILITY_PHRASES):
        return REFLECT_DIAGNOSIS_CAPABILITY_LIMIT
    if "## report" not in low:
        return REFLECT_DIAGNOSIS_NO_REPORT_HEADING
    missing = [s for s in _REQUIRED_SUBSECTIONS if s.lower() not in low]
    if missing:
        return REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE.format(
            missing_subsections=", ".join(f"'{s}'" for s in missing)
        )
    return None


class AgentNode:
    """Base class for ADAS-inspectable LangGraph nodes.

    Subclasses override __call__ to define routing topology.
    inspect.getsource(MyNode.__call__) reads the full routing logic.
    """

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def __call__(self, state: AgenticState) -> Any:
        raise NotImplementedError


class StrategizerNode(AgentNode):
    """Orchestrator: reads Reports, decides next Delegation or Done/Ask."""

    def __init__(
        self,
        adapter: Any,
        name: str,
        outgoing: list[str],
        spec: Any,
        study_dir: Any = None,
        interactive: bool = False,
        max_ask: int = 1,
        worker_adapters: dict | None = None,
    ) -> None:
        super().__init__(adapter)
        self._name = name
        self._outgoing = list(outgoing)
        self._spec = spec
        self._route: dict = {}
        self._study_dir = study_dir
        self._interactive = interactive
        self._max_ask = max_ask
        self._ask_count = 0
        self._current_notes_dir: Path | None = None
        # Parallel delegation registry: id → {"status", "result", "evals"}
        self._worker_adapters: dict[str, Any] = worker_adapters or {}
        self._registry: dict[str, dict] = {}
        self._registry_lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        # Budget state — set at the start of each __call__ from AgenticState
        self._budget_seconds: float | None = None
        self._run_start: float | None = None
        self.adapter.closure_tools.update(self._build_routing_closures())
        self.adapter.route_watcher = lambda: self._route.get("kind") == "done"

    def _build_routing_closures(self) -> dict:
        """Return routing closure tools that write to self._route."""
        node = self
        route = self._route
        outgoing = self._outgoing
        study_dir = self._study_dir
        interactive = self._interactive

        def Delegate(target: str, intent: str, expected_report: str) -> str:
            """Fire a task concurrently; returns a TASK-xxxxxxxx ID immediately."""
            if target not in outgoing:
                return (
                    f"ERROR: unknown target {target!r}."
                    f" Valid targets: {outgoing}"
                )
            worker_template = node._worker_adapters.get(target)
            if worker_template is None:
                return (
                    f"ERROR: no worker adapter for target {target!r}."
                    f" Available: {list(node._worker_adapters)}"
                )

            # Build task message (same as previous sequential logic)
            edge = node._spec.edge(node._name, target)
            preamble = edge.preamble if edge else ""
            task_msg = intent
            if expected_report:
                task_msg += (
                    f"\n\n**Required deliverables / acceptance"
                    f" criteria:**\n{expected_report}"
                )
            if preamble:
                task_msg = preamble + "\n\n" + task_msg

            delegation_id = "TASK-" + uuid.uuid4().hex[:8]
            start_time_mono = time.monotonic()

            with node._registry_lock:
                node._registry[delegation_id] = {
                    "status": "Working",
                    "result": None,
                    "evals": 0,
                    "start_time": start_time_mono,
                }

            # Each delegation gets its OWN adapter copy so concurrent
            # delegations to the same target never race on closure_tools (D1/D2).
            # Adapters that don't implement copy() are used as-is (e.g. test stubs).
            worker = worker_template.copy() if hasattr(worker_template, "copy") else worker_template

            def _run() -> None:
                evals_box: dict = {"count": 0}

                def ReportEvals(count: int) -> str:
                    """Report the number of function evaluations used."""
                    evals_box["count"] = int(count)
                    return f"Recorded {count} evaluations."

                worker.closure_tools["ReportEvals"] = ReportEvals
                try:
                    from .agent_prompts import IMPLEMENTER_REPORT_RETRY_PROMPT

                    messages = [{"role": "user", "content": task_msg}]
                    text = worker.invoke(messages)

                    diagnosis = _classify_response(text)
                    if diagnosis is not None:
                        retry_messages = messages + [
                            {"role": "ai", "content": text},
                            {
                                "role": "user",
                                "content": (
                                    f"{IMPLEMENTER_REPORT_RETRY_PROMPT}"
                                    f"\n\nDiagnosis: {diagnosis}"
                                ),
                            },
                        ]
                        text = worker.invoke(retry_messages)

                    with node._registry_lock:
                        node._registry[delegation_id].update({
                            "status": "Done",
                            "result": text,
                            "evals": evals_box["count"],
                        })
                except Exception:  # noqa: BLE001
                    tb = traceback.format_exc()
                    with node._registry_lock:
                        node._registry[delegation_id].update({
                            "status": "Errored",
                            "result": tb,
                            "evals": evals_box["count"],
                        })

            t = threading.Thread(target=_run, daemon=True, name=delegation_id)
            with node._registry_lock:
                node._threads[delegation_id] = t  # A3: inside lock
            t.start()

            return (
                f"Delegation started. ID: {delegation_id!r}. "
                f"Use GetStatus('{delegation_id}') to poll for completion."
            )

        def GetStatus(delegation_id: str) -> str:
            """Poll a background delegation.

            Returns one of:
              'Working'                 — task still running
              'Done\\n\\n<full report>' — completed successfully
              'Errored:\\n<traceback>'  — task raised an exception; read the
                                         traceback, revise intent, re-delegate
            """
            with node._registry_lock:
                entry = dict(node._registry.get(delegation_id, {}))
            if not entry:
                known = list(node._registry)
                return (
                    f"ERROR: unknown delegation ID {delegation_id!r}. "
                    f"Known IDs: {known}"
                )
            status = entry["status"]
            if status == "Working":
                # Lazy timeout: 75% of the run's total time budget.
                # No budget → no timeout (B2 fix, but respects user's config).
                budget = node._budget_seconds
                if budget is not None and budget > 0:
                    timeout = 0.75 * budget
                    elapsed = time.monotonic() - entry["start_time"]
                    if elapsed > timeout:
                        with node._registry_lock:
                            node._registry[delegation_id].update({
                                "status": "Errored",
                                "result": (
                                    f"Timeout: delegation exceeded {timeout:.0f}s "
                                    f"(75% of {budget:.0f}s run budget; "
                                    f"elapsed {elapsed:.0f}s)."
                                ),
                            })
                        return (
                            f"Errored:\nTimeout: delegation exceeded {timeout:.0f}s "
                            f"(75% of run budget). Re-delegate with a simpler "
                            "or more focused task."
                        )
                return "Working"
            if status == "Done":
                return f"Done\n\n{entry['result']}"
            return f"Errored:\n{entry['result']}"

        def Done(summary: str) -> str:
            """Signal end of run with a summary of findings.

            Refuses if any delegation is still Working — call GetStatus()
            on all pending delegations first.
            """
            with node._registry_lock:
                pending = [
                    did for did, e in node._registry.items()
                    if e["status"] == "Working"
                ]
            if pending:
                return (
                    f"ERROR: {len(pending)} delegation(s) still running: "
                    f"{pending}. "
                    "Call GetStatus() on each and wait for 'Done' or "
                    "'Errored' before calling Done()."
                )
            route["kind"] = "done"
            route["summary"] = summary
            return "Run complete."

        def Ask(question: str) -> str:
            """Ask the human operator a question and wait for input."""
            if node._ask_count >= node._max_ask:
                return (
                    f"Ask limit reached ({node._max_ask} allowed). "
                    "Proceed autonomously with the information you have."
                )
            node._ask_count += 1
            if interactive:
                print(f"\n[Strategizer] {question}\nAnswer: ", end="", flush=True)
                return input()
            # Non-interactive: auto-respond so run proceeds autonomously.
            return (
                "No human operator is present. Proceed autonomously "
                "using only information available in the problem statement "
                "and files in the study directory."
            )

        def WriteMarkdown(path: str, body: str) -> str:
            """Write a Markdown (.md) file to strategizer_notes/."""
            notes_dir = node._current_notes_dir
            if notes_dir is None:
                return "ERROR: notes_dir not set (run_dir missing from state)."
            bare = Path(path).name
            if not bare.endswith(".md"):
                bare = bare + ".md"
            target = Path(notes_dir) / bare
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
            return f"Written: {target}"

        def ReadNote(path: str) -> str:
            """Read a file from the study directory."""
            if study_dir is None:
                return "ERROR: study_dir not set."
            target = Path(study_dir) / path
            if not target.exists():
                return f"NOT FOUND: {target}"
            return target.read_text(encoding="utf-8")

        return {
            "Delegate": Delegate,
            "GetStatus": GetStatus,
            "Done": Done,
            "Ask": Ask,
            "WriteMarkdown": WriteMarkdown,
            "ReadNote": ReadNote,
        }

    def _missing_deliverables(self, state: AgenticState) -> list[str]:
        """Return required deliverable paths not present yet."""
        required = state.get("required_deliverables") or []
        study_dir = Path(state.get("study_dir", "."))
        return [p for p in required if not (study_dir / p).exists()]

    def __call__(self, state: AgenticState) -> Any:
        import time

        from langchain_core.messages import AIMessage, HumanMessage
        from langgraph.graph import END
        from langgraph.types import Command, interrupt

        # Update notes_dir from current state run_dir
        run_dir = state.get("run_dir")
        if run_dir:
            self._current_notes_dir = Path(run_dir) / "strategizer_notes"

        # Soft budget warnings — appended to context, run is NOT stopped.
        # 95%: early warning, start wrapping up.
        # 100%: final warning with 5% cleanup window remaining.
        budget_warnings: list[dict] = []
        budget = state.get("budget_seconds")
        start = state.get("start_time")
        # Store on node so GetStatus() can compute delegation timeout
        self._budget_seconds = budget
        self._run_start = start
        if budget is not None and start is not None:
            elapsed = time.time() - start
            pct = elapsed / budget
            if pct >= 1.0:
                cleanup_remaining = max(0.0, budget * 1.05 - elapsed)
                budget_warnings.append({
                    "role": "user",
                    "content": (
                        f"CRITICAL: time budget fully consumed "
                        f"({elapsed:.0f}s / {budget:.0f}s). "
                        f"You have ~{cleanup_remaining:.0f}s (5% cleanup window) "
                        "to call Done(). Do not start new delegations."
                    ),
                })
            elif pct >= 0.95:
                budget_warnings.append({
                    "role": "user",
                    "content": (
                        f"Warning: time budget at {pct*100:.0f}% "
                        f"({elapsed:.0f}s / {budget:.0f}s). "
                        "Begin wrapping up — call Done() soon."
                    ),
                })

        eval_budget = state.get("eval_budget")
        evals_used = state.get("evals_used", 0)
        if eval_budget is not None and evals_used >= eval_budget:
            budget_warnings.append({
                "role": "user",
                "content": (
                    f"Warning: eval budget exceeded"
                    f" ({evals_used} used / {eval_budget} budget)."
                    f" Do not run further evaluations."
                ),
            })

        # A1/A2: reset per-turn state so a reused node starts clean each call
        self._route.clear()
        self._ask_count = 0
        with self._registry_lock:
            self._registry.clear()
            self._threads.clear()

        messages = _to_adapter_messages(state["messages"]) + budget_warnings
        text = self.adapter.invoke(messages)
        ai_msg = AIMessage(content=text)

        route = self._route
        # "done" or no routing tool — enforce deliverables before accepting
        missing = self._missing_deliverables(state)
        if missing:
            missing_list = "\n".join(f"- {p}" for p in missing)
            return Command(
                goto=self._name,
                update={
                    "messages": [
                        ai_msg,
                        HumanMessage(content=(
                            "Run cannot complete: the following required"
                            " deliverables are missing from the"
                            f" workspace:\n{missing_list}\n"
                            "Please delegate their creation before"
                            " calling Done."
                        )),
                    ],
                },
            )

        # Accumulate delegation counts and evals from registry
        with self._registry_lock:
            total_new = len(self._registry)
            evals_new = sum(e["evals"] for e in self._registry.values())

        summary = route.get("summary") or text
        return Command(
            goto=END,
            update={
                "messages": [ai_msg],
                "done": True,
                "last_report": summary,
                "total_delegations": state["total_delegations"] + total_new,
                "evals_used": state.get("evals_used", 0) + evals_new,
            },
        )


class ImplementerNode(AgentNode):
    """Worker node: executes tasks, writes Reports, returns to caller."""

    def __init__(self, adapter: Any) -> None:
        super().__init__(adapter)
        self._evals_reported: dict = {}
        self.adapter.closure_tools.update(self._build_eval_closures())

    def _build_eval_closures(self) -> dict:
        evals = self._evals_reported

        def ReportEvals(count: int) -> str:
            """Report the number of function evaluations used in this task."""
            evals["count"] = int(count)
            return f"Recorded {count} evaluations."

        return {"ReportEvals": ReportEvals}

    def __call__(self, state: AgenticState) -> Any:
        from langchain_core.messages import AIMessage
        from langgraph.types import Command

        from .agent_prompts import IMPLEMENTER_REPORT_RETRY_PROMPT

        self._evals_reported.clear()
        messages = _to_adapter_messages(state["messages"])
        text = self.adapter.invoke(messages)

        diagnosis = _classify_response(text)
        if diagnosis is not None:
            # One retry with correction prompt
            retry_messages = messages + [
                {"role": "ai", "content": text},
                {
                    "role": "user",
                    "content": (
                        f"{IMPLEMENTER_REPORT_RETRY_PROMPT}"
                        f"\n\nDiagnosis: {diagnosis}"
                    ),
                },
            ]
            text = self.adapter.invoke(retry_messages)

        ai_msg = AIMessage(content=text)
        evals_delta = self._evals_reported.get("count", 0)
        return_to = state.get("return_to")
        return Command(
            goto=return_to,
            update={
                "messages": [ai_msg],
                "last_report": text,
                "evals_used": state.get("evals_used", 0) + evals_delta,
            },
        )
